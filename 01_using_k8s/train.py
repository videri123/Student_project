import pandas as pd
import numpy as np  
import logging
import os
import boto3
import pickle
import shutil
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression   
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Define local paths for data and model
local_data_path = "data/StudentPerformanceFactors.csv"
local_model_path = "model/model.pkl"

#load the dataset into a Pandas DataFrame and Create data folder if it doesn't exist
os.makedirs(os.path.dirname(local_data_path), exist_ok=True)    

# Load dataset
df = pd.read_csv(local_data_path, encoding="latin1", low_memory=False)



# Define features (X) and labels (y)
X = df[[     "Hours_Studied",
    "Sleep_Hours",
    "Attendance" ]].values
y = df['Exam_Score'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# Initialize the Linear Regression model
lr = LinearRegression()
model = lr.fit(X_train, y_train)

# Create and train models.
# rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
# rf.fit(X_train, y_train)

# Save the model using pickle
os.makedirs(os.path.dirname(local_model_path), exist_ok=True)
with open(local_model_path, 'wb') as f:
    pickle.dump(model, f)
logging.info("Model saved locally successfully")

# Predict on the test set
y_pred = lr.predict(X_test)

# Define a function to calculate evaluation metrics
def eval_metrics_rmse(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    return rmse 

def eval_metrics_r2(actual, pred):
    r2 = r2_score(actual, pred)
    return r2

# Evaluate the model
rmse =  eval_metrics_rmse(y_test, y_pred)
r2 = eval_metrics_r2(y_test, y_pred)

print("Model evaluation metrics:")
print(f"RMSE: {rmse}")
print(f"R2 Score: {r2}")
