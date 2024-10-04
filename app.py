from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
import numpy as np
import pickle
import logging
from tensorflow.keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware
from functools import lru_cache
import os

# Initialize FastAPI app
app = FastAPI(
    title="Maintenance Issue Prediction API",
    description="Predicts time to resolve maintenance issues based on description, severity, and other features.",
    version="2.1.0"
)

# Setup logging
logging.basicConfig(
    filename="app.log",
    filemode="a",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables (paths)
MODEL_PATH = os.getenv('MODEL_PATH', 'issue_predictor_model.keras')
SCALER_PATH = os.getenv('SCALER_PATH', 'scaler.pkl')
TFIDF_PATH = os.getenv('TFIDF_PATH', 'tfidf.pkl')

# Setup CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for dev)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and preprocessing assets on startup (cached to avoid reloading)
@lru_cache()
def load_assets():
    try:
        model = load_model(MODEL_PATH)
        with open(SCALER_PATH, 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        with open(TFIDF_PATH, 'rb') as tfidf_file:
            tfidf = pickle.load(tfidf_file)
        logger.info("Model, scaler, and TFIDF loaded successfully.")
        return model, scaler, tfidf
    except Exception as e:
        logger.error(f"Error loading assets: {e}")
        raise HTTPException(status_code=500, detail="Failed to load assets.")

# Request validation using Pydantic
class MaintenanceIssue(BaseModel):
    description: str
    severity: int
    occurrence: int
    detection: int

# Calculate RPN dynamically
def calculate_rpn(severity: int, occurrence: int, detection: int) -> int:
    return severity * occurrence * detection

# Generate recommendation based on severity, RPN, and downtime
def generate_recommendation(issue: MaintenanceIssue, predicted_time: float, weighted_time: float, rpn: int) -> str:
    if issue.severity >= 8:
        return (
            f"The issue has a high severity level ({issue.severity}/10), demanding immediate attention. "
            f"Predicted resolution time is {predicted_time:.2f} hours, and weighted resolution time is {weighted_time:.2f} hours. "
            "Consider assigning experienced personnel to mitigate downtime."
        )
    elif rpn > 200:
        return (
            f"High RPN detected ({rpn}/1000), indicating a critical risk. "
            f"The issue should be resolved in {predicted_time:.2f} hours. Ensure preventive measures are in place "
            "to avoid recurrence."
        )
    else:
        return (
            f"Expected resolution time: {predicted_time:.2f} hours. "
            f"Weighted resolution time: {weighted_time:.2f} hours. Ensure efficient resource allocation."
        )

# Predict maintenance issue resolution time
@app.post("/predict/", response_model=dict, status_code=status.HTTP_200_OK)  # Set explicit status code
async def predict_issue(issue: MaintenanceIssue):
    try:
        logger.info(f"Processing issue: {issue.description}")

        # Load assets once
        model, scaler, tfidf = load_assets()

        # Calculate RPN
        rpn = calculate_rpn(issue.severity, issue.occurrence, issue.detection)
        logger.info(f"Calculated RPN: {rpn}")

        # Transform input description using the TFIDF vectorizer
        description_vec = tfidf.transform([issue.description]).toarray()

        # Ensure the TF-IDF vector has the correct shape (None, 50)
        expected_tfidf_shape = 50
        if description_vec.shape[1] < expected_tfidf_shape:
            # Pad the vector with zeros if it has fewer features than expected
            description_vec = np.pad(description_vec, ((0, 0), (0, expected_tfidf_shape - description_vec.shape[1])), 'constant')
        elif description_vec.shape[1] > expected_tfidf_shape:
            # Truncate the vector if it has more features than expected
            description_vec = description_vec[:, :expected_tfidf_shape]

        # Combine numeric features and scale them
        numeric_features = np.array([[issue.severity, issue.occurrence, issue.detection]])
        numeric_features_scaled = scaler.transform(numeric_features)

        # Model prediction
        prediction = model.predict([description_vec, numeric_features_scaled])

        if prediction is None or len(prediction) == 0:
            raise ValueError("Model returned an empty prediction")

        predicted_time = float(prediction[0][0])

        # Log the input and output of the prediction
        logger.info(f"Input: {issue.dict()}, Predicted Time: {predicted_time:.2f}")

        # Apply a default frequency weight since issue_frequency is removed
        frequency_weight = 1  # Default weight since issue_frequency is not included
        weighted_time = predicted_time * frequency_weight

        # Generate recommendation
        recommendation = generate_recommendation(issue, predicted_time, weighted_time, rpn)

        logger.info(f"Prediction successful: {predicted_time} hours")

        # Explicitly return HTTP 200 OK with a message
        return {
            "message": "Prediction successful",  # Adding a success message
            "predicted_time": round(predicted_time, 2),
            "weighted_time": round(weighted_time, 2),
            "frequency_weight": frequency_weight,
            "rpn": rpn,
            "recommended_solution": recommendation
        }
    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

# Startup event to load model and scaler
@app.on_event("startup")
async def startup_event():
    try:
        load_assets()
        logger.info("Model, scaler, and TFIDF loaded successfully.")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise HTTPException(status_code=500, detail="Failed to load assets.")

# FastAPI root endpoint
@app.get("/")
async def main():
    return {"message": "Welcome to the Maintenance Issue Prediction API!"}

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
