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


# Logging
logging.basicConfig(level=logging.INFO)

# MLflow
mlflow.set_tracking_uri("http://localhost:5050")
mlflow.set_experiment("StudentPerformanceFactors_exp_v4")
mlflow.autolog()

# S3 config
bucket = "babji-mlops"
data_s3_path = "data/StudentPerformanceFactors.csv"
model_s3_path = "model/model.pkl"

local_data_path = "data/StudentPerformanceFactors.csv"
local_model_path = "model/model.pkl"

# AWS credentials
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

# Check AWS credentials
if aws_access_key_id and aws_secret_access_key:
    logging.info("AWS Access Key and Secret Key have been retrieved successfully.")
    logging.info("AWS Access Key ID: %s", aws_access_key_id)
    logging.info(
        "AWS Secret Access Key: %s",
        aws_secret_access_key[:4] + "*" * 16 + aws_secret_access_key[-4:],
    )
else:
    raise EnvironmentError("AWS Access Key or Secret Key not set properly.")

# Create S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
)

# Create directories
os.makedirs(os.path.dirname(local_data_path), exist_ok=True)
os.makedirs(os.path.dirname(local_model_path), exist_ok=True)

# Download dataset from S3
logging.info("Downloading dataset from S3...")
s3_client.download_file(bucket, data_s3_path, local_data_path)
logging.info("Dataset downloaded successfully.")

# Load dataset
df = pd.read_csv(local_data_path, low_memory=False)

# Define features and target
X = df[["Hours_Studied", "Sleep_Hours", "Attendance"]]
y = df["Exam_Score"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

with mlflow.start_run():
    # Define two regression algorithms
    algorithms = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
            max_features=2,
            random_state=42,
        ),
    }

    # Train both models
    logging.info("Training models...")
    lr_model = algorithms["Linear Regression"].fit(X_train, y_train)
    rf_model = algorithms["Random Forest"].fit(X_train, y_train)

    # Evaluation functions
    def eval_rmse(actual, pred):
        return np.sqrt(mean_squared_error(actual, pred))

    def eval_r2(actual, pred):
        return r2_score(actual, pred)

    # Predictions
    lr_pred = lr_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)

    # Evaluate both models
    lr_r2 = eval_r2(y_test, lr_pred)
    rf_r2 = eval_r2(y_test, rf_pred)

    lr_rmse = eval_rmse(y_test, lr_pred)
    rf_rmse = eval_rmse(y_test, rf_pred)

    # Log performance
    logging.info(f"Linear Regression --> R2: {lr_r2}, RMSE: {lr_rmse}")
    logging.info(f"Random Forest --> R2: {rf_r2}, RMSE: {rf_rmse}")

    # Log metrics to MLflow
    mlflow.log_metric("lr_r2", lr_r2)
    mlflow.log_metric("lr_rmse", lr_rmse)
    mlflow.log_metric("rf_r2", rf_r2)
    mlflow.log_metric("rf_rmse", rf_rmse)

    # Select best model based on R2 score
    if rf_r2 > lr_r2:
        best_model = rf_model
        best_model_name = "Random Forest"
    else:
        best_model = lr_model
        best_model_name = "Linear Regression"

    logging.info(f"Best model selected: {best_model_name}")
    mlflow.log_param("best_model_name", best_model_name)

    # Final prediction using best model
    y_pred = best_model.predict(X_test)

    # Final metrics
    final_rmse = eval_rmse(y_test, y_pred)
    final_r2 = eval_r2(y_test, y_pred)

    logging.info(f"Final Model RMSE: {final_rmse}")
    logging.info(f"Final Model R2: {final_r2}")

    mlflow.log_metric("final_rmse", final_rmse)
    mlflow.log_metric("final_r2", final_r2)

    # Save best model locally
    with open(local_model_path, "wb") as f:
        pickle.dump(best_model, f)

    logging.info("Best model saved locally successfully.")

    # Upload best model to S3
    try:
        logging.info("Uploading best model to S3...")
        s3_client.upload_file(local_model_path, bucket, model_s3_path)
        logging.info("Best model uploaded to S3 successfully.")
    except Exception as e:
        logging.error(f"Failed to upload model to S3: {e}")
        raise

# Log successful completion
logging.info("Pipeline completed successfully.")