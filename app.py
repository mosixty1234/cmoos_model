# fastapi_app.py

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
import numpy as np
import pickle
import logging
from tensorflow.keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app with metadata
app = FastAPI(
    title="Maintenance Issue Prediction API",
    description="Predicts time to resolve maintenance issues based on description, severity, and other features.",
    version="2.0.0"
)

# Load environment variables (production settings)
MODEL_PATH = 'issue_predictor_model.keras'
SCALER_PATH = 'scaler.pkl'
TFIDF_PATH = 'tfidf.pkl'

# CORS middleware (for local development, adjust for production)
origins = ["http://127.0.0.1:8080"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["Authorization", "Content-Type"],
)

# Setup enhanced logging: log to file for production
logging.basicConfig(
    filename="app.log",
    filemode="a",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Preload model and assets during startup
@app.on_event("startup")
async def startup_event():
    global model, scaler, tfidf
    try:
        model, scaler, tfidf = load_assets()
        logger.info("Model, scaler, and TFIDF loaded successfully.")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise HTTPException(status_code=500, detail="Failed to load assets.")

# Load model and other assets
def load_assets():
    try:
        model = load_model(MODEL_PATH)
        with open(SCALER_PATH, 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        with open(TFIDF_PATH, 'rb') as tfidf_file:
            tfidf = pickle.load(tfidf_file)
        return model, scaler, tfidf
    except Exception as e:
        logger.error(f"Error loading assets: {e}")
        raise HTTPException(status_code=500, detail="Failed to load assets.")

# Root endpoint to check API status
@app.get("/")
async def read_root():
    return {"message": "API is up and running!"}

# Health check to monitor API readiness
@app.get("/health/")
async def health_check():
    try:
        model, scaler, tfidf = load_assets()
        return {"status": "healthy"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="API health check failed")

# Fix for favicon.ico error
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

# Request validation using Pydantic
class MaintenanceIssue(BaseModel):
    description: str
    severity: float = Field(..., gt=0, lt=11, description="Severity must be between 1 and 10")
    total_downtime: float = Field(..., gt=0, description="Total downtime must be positive")
    oee: float = Field(..., gt=0, lt=1.1, description="OEE must be between 0 and 1")
    issue_frequency: int = Field(..., ge=0, description="Issue frequency must be non-negative")

# Predict maintenance issue resolution time
@app.post("/predict/")
async def predict_issue(issue: MaintenanceIssue):
    try:
        logger.info(f"Processing issue: {issue.description}")
        
        # Transform input description using the TFIDF vectorizer
        description_vec = tfidf.transform([issue.description]).toarray()
        
        # Combine numeric features and scale them
        numeric_features = np.array([[issue.severity, issue.total_downtime, issue.oee]])
        numeric_features_scaled = scaler.transform(numeric_features)

        # Model prediction
        prediction = model.predict([description_vec, numeric_features_scaled])
        
        if prediction is None or len(prediction) == 0:
            raise ValueError("Model returned an empty prediction")

        predicted_time = float(prediction[0][0])

        # Apply weighting based on issue frequency
        frequency_weight = 1 + (issue.issue_frequency / 10)
        weighted_time = predicted_time * frequency_weight

        # Generate recommendation
        recommendation = generate_recommendation(issue, predicted_time)

        logger.info(f"Prediction successful: {predicted_time} hours")

        return {
            "predicted_time": predicted_time,
            "weighted_time": weighted_time,
            "frequency_weight": frequency_weight,
            "recommended_solution": recommendation
        }
    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

# Generate recommendation based on input data
def generate_recommendation(issue: MaintenanceIssue, predicted_time: float) -> str:
    if issue.severity >= 8:
        return (
            f"High severity ({issue.severity}/10). Immediate attention required, "
            f"predicted resolution time: {predicted_time:.2f} hours. "
            "Assign senior staff to prevent extended downtime."
        )
    elif issue.oee < 0.7:
        return (
            f"Low OEE ({issue.oee:.2f}). Consider preventive maintenance in addition to "
            f"resolving the issue in {predicted_time:.2f} hours. "
            "Improving efficiency could reduce future downtimes."
        )
    elif issue.total_downtime > 5:
        return (
            f"Significant downtime ({issue.total_downtime:.2f} hours). Allocate resources efficiently "
            f"to resolve within {predicted_time:.2f} hours and minimize impact."
        )
    else:
        return f"Expected resolution time: {predicted_time:.2f} hours. Ensure proper resource allocation."

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
