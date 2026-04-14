import os
import boto3
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# =========================
# CONFIG
# =========================
bucket = "babji-mlops"
model_s3_path = "model/model.pkl"
local_model_path = "model/model.pkl"

# =========================
# AWS SETUP
# =========================
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

if not aws_access_key_id or not aws_secret_access_key:
    raise EnvironmentError("AWS credentials not set!")

s3_client = boto3.client(
    "s3",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
)

# =========================
# CREATE FOLDERS
# =========================
os.makedirs("model", exist_ok=True)

# =========================
# LOAD DATA (LOCAL)
# =========================
print("Loading dataset from local folder...")
df = pd.read_csv("data/StudentPerformanceFactors.csv")

# =========================
# SIMPLE TRAINING
# =========================
X = df.drop("Exam_Score", axis=1)
y = df["Exam_Score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

print("Model training complete!")

# =========================
# SAVE MODEL
# =========================
with open(local_model_path, "wb") as f:
    pickle.dump(model, f)

print("Model saved locally!")

# =========================
# UPLOAD MODEL TO S3
# =========================
print("Uploading model to S3...")

s3_client.upload_file(local_model_path, bucket, model_s3_path)

print("Model uploaded successfully!")