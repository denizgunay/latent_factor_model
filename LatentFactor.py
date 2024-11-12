import pandas as pd


class LatentFactorModel:
    """
    Latent Factor Model for collaborative filtering using Singular Value Decomposition (SVD).
    """

    def __init__(self):
        """
        Initialize the LatentFactorModel class with default values.
        """
        self.params = None
        self.model = None
        self.rmse = None
        self.mse = None
        self.rating_col_name = None
        self.lr_all = None
        self.reg_all = None
        self.n_epochs = None
        self.n_factors = None

    def sampler(self, dataframe, item_id_col_name, sample_ratio=80):
        """
        Sample the dataframe based on item popularity.

        Parameters:
        - dataframe: The input DataFrame containing user-item interactions.
        - item_id_col_name: Name of the column representing item IDs.
        - sample_ratio: Percentage of items to keep based on popularity (default: 80).

        Returns:
        - sample_df: Sampled DataFrame based on item popularity.
        """
        if not (0 < sample_ratio < 100):
            return dataframe
        item_counts = pd.DataFrame(dataframe[item_id_col_name].value_counts())
        item_counts.rename(columns={item_id_col_name: "counts"}, inplace=True)
        item_counts = item_counts.rename_axis(item_id_col_name)
        item_counts = item_counts.reset_index()
        item_counts["cum_sum"] = (
            100 * item_counts["counts"].cumsum() / item_counts["counts"].sum()
        )
        sample_df = dataframe[
            dataframe[item_id_col_name].isin(
                item_counts[item_counts.cum_sum <= sample_ratio][item_id_col_name]
            )
        ]
        sample_df = sample_df.reset_index(drop=True)
        return sample_df

    def cross_validate(
        self,
        sample_dataframe,
        user_id_col_name,
        item_id_col_name,
        rating_col_name,
        params_grid,
        cv=3,
        rating_scale=(1, 5),
        scoring="rmse",
    ):
        """
        Perform cross-validation to find the best hyperparameters using GridSearch.

        Parameters:
        - sample_dataframe: The sampled DataFrame for cross-validation.
        - user_id_col_name: Name of the column representing user IDs.
        - item_id_col_name: Name of the column representing item IDs.
        - rating_col_name: Name of the column representing ratings.
        - params_grid: Dictionary of hyperparameter grids for GridSearch.
        - cv: Number of cross-validation folds (default: 3).
        - rating_scale: Tuple representing the rating scale (min, max) (default: (1,5)).
        - scoring: Scoring metric to optimize during cross-validation (default: "rmse").

        Returns:
        - None
        """
        from surprise.model_selection import GridSearchCV
        from surprise import Reader, SVD, Dataset

        self.rating_col_name = rating_col_name
        reader = Reader(rating_scale=rating_scale)
        data = Dataset.load_from_df(
            sample_dataframe[[user_id_col_name, item_id_col_name, rating_col_name]],
            reader,
        )

        grid_search = GridSearchCV(
            SVD,
            params_grid,
            measures=["rmse", "mse"],
            cv=cv,
            n_jobs=-1,
            joblib_verbose=True,
        )

        grid_search.fit(data)
        self.params = grid_search.best_params[scoring]
        self.rmse = grid_search.best_score["rmse"]
        self.mse = grid_search.best_score["mse"]
        print("Parameters:")
        print(self.params)
        print(f"Best {scoring}:")
        if scoring == "rmse":
            print(self.rmse)
        elif scoring == "mse":
            print(self.mse)

    def fit_predict(
        self,
        dataframe,
        user_id_col_name,
        item_id_col_name,
        rating_col_name,
        rating_scale=(1, 5),
        params=None,
    ):
        """
        Train the Latent Factor Model, make predictions, and evaluate the model on the input dataframe.

        Parameters:
        - dataframe: The input DataFrame containing user-item interactions.
        - user_id_col_name: Name of the column representing user IDs.
        - item_id_col_name: Name of the column representing item IDs.
        - rating_col_name: Name of the column representing ratings.
        - rating_scale: Tuple representing the rating scale (min, max) (default: (1,5)).
        - params: Optional hyperparameters for the SVD model (default: None).

        Returns:
        - predictions: DataFrame containing predicted ratings and model evaluation metrics.
        """

        from surprise import Reader, SVD, Dataset

        self.rating_col_name = rating_col_name
        reader = Reader(rating_scale=rating_scale)
        data = Dataset.load_from_df(
            dataframe[[user_id_col_name, item_id_col_name, rating_col_name]], reader
        )
        data = data.build_full_trainset()

        if params != None:
            self.params = params
            svd_model = SVD(**params)
        else:
            if self.params != None:
                svd_model = SVD(**self.params)
            else:
                svd_model = SVD()

        self.lr_all = svd_model.lr_qi
        self.reg_all = svd_model.reg_qi
        self.n_epochs = svd_model.n_epochs
        self.n_factors = svd_model.n_factors
        self.params = {
            "n_factors": svd_model.n_factors,
            "n_epochs": svd_model.n_epochs,
            "lr_all": svd_model.lr_qi,
            "reg_all": svd_model.reg_qi,
        }
        svd_model.fit(data)
        self.model = svd_model
        predictions = []
        for index, row in dataframe[[user_id_col_name, item_id_col_name]].iterrows():
            user = row[user_id_col_name]
            item = row[item_id_col_name]
            prediction = svd_model.predict(uid=user, iid=item, verbose=False)
            predictions.append(
                [
                    prediction[0],
                    prediction[1],
                    dataframe.loc[index, rating_col_name],
                    prediction[3],
                    prediction[4],
                ]
            )

        import numpy as np

        mse = np.mean(
            [float((true_r - est) ** 2) for (_, _, true_r, est, _) in predictions]
        )
        rmse = np.sqrt(mse)
        self.rmse = rmse
        self.mse = mse

        predictions = pd.DataFrame(predictions)
        predictions.columns = [
            user_id_col_name,
            item_id_col_name,
            rating_col_name,
            "estimated_" + rating_col_name,
            "details",
        ]
        return predictions

    def recommend(self, dataframe, user_id, user_id_col_name, item_id_col_name):
        """
        Generate item recommendations for a given user.

        Parameters:
        - dataframe: The input DataFrame containing user-item interactions.
        - user_id: ID of the target user for recommendations.
        - user_id_col_name: Name of the column representing user IDs.
        - item_id_col_name: Name of the column representing item IDs.

        Returns:
        - rec_df: DataFrame containing recommended items for the specified user.
        """
        watched_items = dataframe[dataframe[user_id_col_name] == user_id][
            item_id_col_name
        ].values
        unique_items = dataframe[item_id_col_name].unique()
        unwatched_items = list(set(unique_items) - set(watched_items))
        rec_list = []
        for item_id in unwatched_items:
            prediction = self.model.predict(uid=user_id, iid=item_id, verbose=False)
            rec_list.append(prediction)
        rec_list = pd.DataFrame(rec_list).sort_values("est", ascending=False)
        est_col_name = "estimated_" + self.rating_col_name
        rec_list.rename(columns={"iid": "movieId", "est": est_col_name}, inplace=True)
        rec_list = rec_list[["movieId", est_col_name]].reset_index(drop=True)
        dataframe = dataframe.drop_duplicates(subset=[item_id_col_name]).reset_index(
            drop=True
        )
        dataframe = dataframe.drop([user_id_col_name, self.rating_col_name], axis=1)
        rec_df = pd.merge(rec_list, dataframe, on=item_id_col_name, how="left")
        return rec_df

    def reset(self):
        """
        Reset all attributes of the LatentFactorModel instance to their initial values.
        """
        self.params = None
        self.model = None
        self.rmse = None
        self.mse = None
        self.rating_col_name = None
        self.lr_all = None
        self.reg_all = None
        self.n_epochs = None
        self.n_factors = None
