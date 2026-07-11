# California House Price Prediction

A machine learning pipeline that predicts median house values in California using the classic *California Housing dataset*. The project covers stratified train/test splitting, a reusable preprocessing pipeline (imputation, scaling, one-hot encoding), and model training/inference with a persisted RandomForestRegressor.

## Overview

- Loads housing.csv and creates an *income-stratified* train/test split (via StratifiedShuffleSplit on binned median_income) so the split reflects the real income distribution instead of a random sample.
- Builds a ColumnTransformer pipeline:
  - *Numerical features* → median imputation + standard scaling
  - *Categorical feature* (ocean_proximity) → one-hot encoding
- Trains a RandomForestRegressor, which outperformed Linear Regression and Decision Tree baselines.
- Persists the trained *model* and *pipeline* with joblib (model.pkl, pipeline.pkl).
- On subsequent runs, skips training and instead runs *inference* on input.csv, writing predictions to output.csv.

## Tech Stack

- Python
- Pandas
- NumPy
- scikit-learn (Pipeline, ColumnTransformer, RandomForestRegressor)
- joblib (model persistence)

## Project Structure

├── housing.csv            # Source dataset
├── input.csv              # Hold-out set generated from the stratified split (used for inference)
├── output.csv             # Predictions written after inference
├── main.py                # Training + inference script
├── main_old.py            # Earlier version of the script
└── README.md

## How It Works

1. First run (no model.pkl present):
   - Reads housing.csv, creates an income_cat column by binning median_income.
   - Performs a stratified 80/20 split; the test portion is saved as input.csv.
   - Fits the preprocessing pipeline and trains the RandomForestRegressor on the training portion.
   - Saves model.pkl and pipeline.pkl.
2. Subsequent runs (model already exists):
   - Loads the saved model and pipeline.
   - Transforms input.csv and generates predictions.
   - Writes results to output.csv.

## Usage

pip install pandas numpy scikit-learn joblib

python main.py

Run it once to train (produces model.pkl, pipeline.pkl, input.csv), then run it again to perform inference and generate output.csv.

## Future Improvements

- Add hyperparameter tuning (GridSearchCV / RandomizedSearchCV) for the Random Forest.
- Add cross-validation metrics/reporting to the console output.
- Add a requirements.txt and simple CLI args (e.g., custom input/output paths).
- Wrap inference in a simple API (Flask/FastAPI) for real-time predictions.

## License

MIT
