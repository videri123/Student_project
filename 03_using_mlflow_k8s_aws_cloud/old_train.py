import os
import logging
import pickle

import mlflow
import numpy as np
import pandas as pd

from mlflow import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Logging
logging.basicConfig(level=logging.INFO)

# MLflow setup
mlflow.set_tracking_uri("http://localhost:5050")
mlflow.set_experiment("StudentPerformanceFactors_exp_v3")
mlflow.autolog()

# Define local paths for data and model
local_data_path = "data/StudentPerformanceFactors.csv"
local_model_path = "model/model.pkl"

# Create folders if needed
os.makedirs(os.path.dirname(local_data_path), exist_ok=True)
os.makedirs(os.path.dirname(local_model_path), exist_ok=True)

# Load dataset
df = pd.read_csv(local_data_path, encoding="latin1", low_memory=False)

# Define features (X) and labels (y)
X = df[["Hours_Studied", "Sleep_Hours", "Attendance"]]
y = df["Exam_Score"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# Define metric functions
def eval_metrics_rmse(actual, pred):
    return np.sqrt(mean_squared_error(actual, pred))

def eval_metrics_r2(actual, pred):
    return r2_score(actual, pred)

# Start MLflow run explicitly
with mlflow.start_run():
    # Initialize and train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save the model using pickle
    with open(local_model_path, "wb") as f:
        pickle.dump(model, f)
    logging.info("Model saved locally successfully.")

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    rmse = eval_metrics_rmse(y_test, y_pred)
    r2 = eval_metrics_r2(y_test, y_pred)

    # Log metrics manually to MLflow
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2_score", r2)

    print("Model evaluation metrics:")
    print(f"RMSE: {rmse}")
    print(f"R2 Score: {r2}")

# Show experiments
client = MlflowClient(tracking_uri="http://localhost:5050")
experiments = client.search_experiments()

for exp in experiments:
    print(exp.experiment_id, exp.name, exp.lifecycle_stage)