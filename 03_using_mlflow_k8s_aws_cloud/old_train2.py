import os
import logging
import pickle

import boto3
import mlflow
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# -------------------------------
# Logging
# -------------------------------
logging.basicConfig(level=logging.INFO)

# -------------------------------
# MLflow configuration
# -------------------------------
mlflow.set_tracking_uri("http://localhost:5050")
mlflow.set_experiment("StudentPerformanceFactors_exp_v4")
mlflow.autolog()

# -------------------------------
# S3 configuration
# -------------------------------
bucket = "babji-mlops"
model_s3_path = "model/model.pkl"
local_model_path = "model/model.pkl"

# -------------------------------
# AWS credentials from environment
# -------------------------------
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

if not aws_access_key_id or not aws_secret_access_key:
    raise EnvironmentError("AWS Access Key or Secret Key not set properly.")

logging.info("AWS credentials retrieved successfully.")

# -------------------------------
# Create S3 client
# -------------------------------
s3_client = boto3.client(
    "s3",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
)

# -------------------------------
# Create local folders
# -------------------------------
os.makedirs("model", exist_ok=True)

# -------------------------------
# Load dataset from local project folder
# -------------------------------
logging.info("Loading dataset from local file...")
df = pd.read_csv("data/StudentPerformanceFactors.csv", low_memory=False)
logging.info("Dataset loaded successfully.")

# -------------------------------
# Define features and target
# -------------------------------
X = df[["Hours_Studied", "Sleep_Hours", "Attendance"]]
y = df["Exam_Score"]

# -------------------------------
# Split data
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

with mlflow.start_run():
    # -------------------------------
    # Define algorithms
    # -------------------------------
    algorithms = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
            max_features=2,
            random_state=42,
        ),
    }

    # -------------------------------
    # Train models
    # -------------------------------
    logging.info("Training models...")
    lr_model = algorithms["Linear Regression"].fit(X_train, y_train)
    rf_model = algorithms["Random Forest"].fit(X_train, y_train)

    # -------------------------------
    # Evaluation functions
    # -------------------------------
    def eval_rmse(actual, pred):
        return np.sqrt(mean_squared_error(actual, pred))

    def eval_r2(actual, pred):
        return r2_score(actual, pred)

    # -------------------------------
    # Predictions
    # -------------------------------
    lr_pred = lr_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)

    # -------------------------------
    # Evaluate models
    # -------------------------------
    lr_r2 = eval_r2(y_test, lr_pred)
    rf_r2 = eval_r2(y_test, rf_pred)

    lr_rmse = eval_rmse(y_test, lr_pred)
    rf_rmse = eval_rmse(y_test, rf_pred)

    logging.info(f"Linear Regression --> R2: {lr_r2}, RMSE: {lr_rmse}")
    logging.info(f"Random Forest --> R2: {rf_r2}, RMSE: {rf_rmse}")

    # -------------------------------
    # Log metrics to MLflow
    # -------------------------------
    mlflow.log_metric("lr_r2", lr_r2)
    mlflow.log_metric("lr_rmse", lr_rmse)
    mlflow.log_metric("rf_r2", rf_r2)
    mlflow.log_metric("rf_rmse", rf_rmse)

    # -------------------------------
    # Select best model
    # -------------------------------
    if rf_r2 > lr_r2:
        best_model = rf_model
        best_model_name = "Random Forest"
    else:
        best_model = lr_model
        best_model_name = "Linear Regression"

    logging.info(f"Best model selected: {best_model_name}")
    mlflow.log_param("best_model_name", best_model_name)

    # -------------------------------
    # Final evaluation
    # -------------------------------
    y_pred = best_model.predict(X_test)

    final_rmse = eval_rmse(y_test, y_pred)
    final_r2 = eval_r2(y_test, y_pred)

    logging.info(f"Final Model RMSE: {final_rmse}")
    logging.info(f"Final Model R2: {final_r2}")

    mlflow.log_metric("final_rmse", final_rmse)
    mlflow.log_metric("final_r2", final_r2)

    # -------------------------------
    # Save best model locally
    # -------------------------------
    with open(local_model_path, "wb") as f:
        pickle.dump(best_model, f)

    logging.info("Best model saved locally successfully.")

    # -------------------------------
    # Upload best model to S3
    # -------------------------------
    try:
        logging.info("Uploading best model to S3...")
        s3_client.upload_file(local_model_path, bucket, model_s3_path)
        logging.info("Best model uploaded to S3 successfully.")
    except Exception as e:
        logging.error(f"Failed to upload model to S3: {e}")
        raise

logging.info("Pipeline completed successfully.")