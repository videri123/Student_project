import os
import boto3
import pandas as pd
import pickle

bucket = "babji-mlops"
model_s3_path = "model/model.pkl"
local_model_path = "model/model.pkl"

aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

s3_client = boto3.client(
    "s3",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
)

os.makedirs("model", exist_ok=True)

df = pd.read_csv("data/StudentPerformanceFactors.csv")

# train model here

with open(local_model_path, "wb") as f:
    pickle.dump(model, f)

s3_client.upload_file(local_model_path, bucket, model_s3_path)