from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
from keras.models import load_model
import joblib
import logging
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras_tuner import RandomSearch

# Initialize FastAPI
app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained models and scalers
model = load_model('issue_predictor_model.keras', compile=False)
tfidf = joblib.load('tfidf_vectorizer.joblib')
scaler = joblib.load('scaler.joblib')

# Define input data schemas using Pydantic
class IssueData(BaseModel):
    description: str
    severity: float = Field(..., gt=0, lt=10)  # Severity between 0 and 10
    total_downtime: float
    oee: float = Field(..., gt=0, lt=1)  # OEE should be between 0 and 1
    issue_frequency: int = Field(..., gt=0)  # Frequency should be greater than 0

# Define function to recommend solution
def recommend_solution(description, tfidf_model, numeric_data, issue_frequency):
    description_vec = tfidf_model.transform([description]).toarray()
    predicted_time = model.predict([description_vec, numeric_data])
    if issue_frequency > 5:
        recommended_solution = "This issue occurs frequently. Consider preventive maintenance or upgrading equipment."
    else:
        recommended_solution = "The issue is rare. Proceed with standard troubleshooting procedures."
    frequency_weight = 1 + (issue_frequency / 10)
    weighted_time = predicted_time[0][0] * frequency_weight
    return recommended_solution, weighted_time

# API endpoint for single issue prediction
@app.post("/predict")
async def predict_issue_fix_time(issue_data: IssueData):
    try:
        issue_vec = tfidf.transform([issue_data.description]).toarray()
        numeric_features = np.array([[issue_data.severity, issue_data.total_downtime, issue_data.oee]])
        numeric_features_scaled = scaler.transform(numeric_features)
        recommended_solution, weighted_time = recommend_solution(
            issue_data.description, tfidf, numeric_features_scaled, issue_data.issue_frequency
        )
        return {
            "issue_description": issue_data.description,
            "recommended_solution": recommended_solution,
            "predicted_time_to_fix": f"{weighted_time:.2f} hours",
            "frequency_weight": 1 + (issue_data.issue_frequency / 10)
        }
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint to handle requests to "/"
@app.get("/")
async def root():
    return {"message": "Welcome to the Maintenance Issue Prediction API"}
