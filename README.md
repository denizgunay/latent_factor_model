# Latent Factor Model for Collaborative Filtering (SVD)

This repository contains a Latent Factor Model implementation for collaborative filtering using Singular Value Decomposition (SVD). The model allows you to perform matrix factorization for recommendation systems, where the goal is to predict missing ratings and recommend items to users based on their past interactions.

## Features

- **Sampling**: A method to sample the dataset based on item popularity.
- **Cross-validation**: Perform cross-validation to tune hyperparameters and select the best model.
- **Training & Prediction**: Train the SVD model, predict ratings, and evaluate the model performance (RMSE, MSE).
- **Recommendation**: Generate item recommendations for a given user.
- **Reset**: Reset the model and its parameters.