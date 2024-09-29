from fastapi import FastAPI, HTTPException, Depends, Form, Response
from pydantic import BaseModel, Field
import numpy as np
import pickle
import logging
from tensorflow.keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware
import os

# Initialize FastAPI app with metadata
app = FastAPI(
    title="Maintenance Issue Prediction API",
    description="Predicts time to resolve maintenance issues based on description, severity, and other features.",
    version="2.0.0"
)

# Load environment variables (for production settings)
MODEL_PATH = 'issue_predictor_model.keras'
SCALER_PATH = 'scaler.pkl'
TFIDF_PATH = 'tfidf.pkl'

# CORS middleware (for local development, adapt for production)
origins = [
    "http://127.0.0.1:8080",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["Authorization", "Content-Type"],
)

# Enhanced logging: log to file for production
logging.basicConfig(
    filename="app.log",
    filemode="a",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load model and other assets
def get_model_assets():
    try:
        model = load_model(MODEL_PATH)
        with open(SCALER_PATH, 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        with open(TFIDF_PATH, 'rb') as tfidf_file:
            tfidf = pickle.load(tfidf_file)
        return model, scaler, tfidf
    except Exception as e:
        logger.error(f"Failed to load model or assets: {e}")
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

# Root endpoint to check API status
@app.get("/")
async def read_root():
    return {"message": "API is up and running!"}

# Health check to monitor API readiness
@app.get("/health/")
async def health_check():
    try:
        model, scaler, tfidf = get_model_assets()
        return {"status": "healthy"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"API health check failed: {str(e)}")

# **Fix for favicon.ico error**: Add a route to handle favicon.ico
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)  # No content for favicon.ico

# Request validation using Pydantic
class MaintenanceIssue(BaseModel):
    description: str
    severity: float = Field(..., gt=0, lt=11, description="Severity should be between 1 and 10")
    total_downtime: float = Field(..., gt=0, description="Total downtime must be positive")
    oee: float = Field(..., gt=0, lt=1.1, description="OEE should be between 0 and 1")
    issue_frequency: int = Field(..., ge=0, description="Issue frequency must be non-negative")

# Predicting maintenance issue
@app.post("/predict/")
async def predict_issue(issue: MaintenanceIssue, assets=Depends(get_model_assets)):
    model, scaler, tfidf = assets
    logger.info(f"Received issue description: {issue.description}")
    
    try:
        # Validate input data
        if not issue.description:
            raise ValueError("Description is empty")
        
        # Transform input data for prediction
        description_vec = tfidf.transform([issue.description]).toarray()
        numeric_features = np.array([[issue.severity, issue.total_downtime, issue.oee]])
        numeric_features_scaled = scaler.transform(numeric_features)

        # Predict timeframe using the pre-trained model
        prediction = model.predict([description_vec, numeric_features_scaled])
        
        if prediction is None or len(prediction) == 0:
            raise ValueError("Model returned an empty prediction")
        
        predicted_time = float(prediction[0][0])
        frequency_weight = 1 + (issue.issue_frequency / 10)
        weighted_time = predicted_time * frequency_weight

        recommended_solution = f"Based on the predicted time of {predicted_time:.2f} hours, consider allocating resources for efficient resolution."

        logger.info(f"Prediction successful: Predicted time: {predicted_time}, Weighted time: {weighted_time}")
        return {
            "predicted_time": predicted_time,
            "weighted_time": weighted_time,
            "frequency_weight": frequency_weight,
            "recommended_solution": recommended_solution
        }

    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        raise HTTPException(status_code=400, detail=f"Invalid input data: {str(ve)}")

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Custom global exception handler for uncaught errors
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return HTTPException(status_code=500, detail="Internal Server Error")
