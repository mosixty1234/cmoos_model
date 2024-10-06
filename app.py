from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, conint
import numpy as np
import pickle
import logging
from tensorflow.keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware
from functools import lru_cache
from fastapi.responses import JSONResponse
import os

# Initialize FastAPI app
app = FastAPI(
    title="Maintenance Issue Prediction API",
    description="Predicts time to resolve maintenance issues based on description, severity, and other features.",
    version="2.4.0"
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
    except FileNotFoundError as fnfe:
        logger.error(f"File not found: {fnfe}")
        raise HTTPException(status_code=500, detail="Model or preprocessing files missing.")
    except Exception as e:
        logger.error(f"Error loading assets: {e}")
        raise HTTPException(status_code=500, detail="Failed to load assets.")

# Request validation using Pydantic
class MaintenanceIssue(BaseModel):
    description: str
    severity: conint(ge=1, le=10) = Field(..., description="Severity (1-10 scale)")
    occurrence: conint(ge=1, le=10) = Field(..., description="Occurrence (1-10 scale)")
    detection: conint(ge=1, le=10) = Field(..., description="Detection (1-10 scale)")

# Calculate RPN dynamically
def calculate_rpn(severity: int, occurrence: int, detection: int) -> int:
    return severity * occurrence * detection

# Generate recommendation based on severity, RPN, and downtime
def generate_recommendation(issue: MaintenanceIssue, predicted_time: float, rpn: int) -> str:
    if issue.severity >= 6:
        recommendation = "High severity issue; urgent response required."
    elif rpn > 300:
        recommendation = "Critical issue due to high RPN; prioritize repair."
    else:
        recommendation = "Issue manageable; proceed with routine fix."
    
    # Tailor additional advice based on predicted time
    if predicted_time > 6:
        recommendation += " Allocate extra resources for longer repair times."
    
    return recommendation

# Helper function to convert predicted time to hours or minutes
def format_time(predicted_time: float) -> str:
    if predicted_time < 1:
        # Convert hours to minutes if less than 1 hour
        minutes = predicted_time * 60
        return f"{round(minutes, 2)} minutes"
    else:
        # Keep time in hours if it's 1 hour or more
        return f"{round(predicted_time, 2)} hours"

@app.post("/predict/", response_model=dict, status_code=status.HTTP_200_OK)
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
            description_vec = np.pad(description_vec, ((0, 0), (0, expected_tfidf_shape - description_vec.shape[1])), 'constant')
        elif description_vec.shape[1] > expected_tfidf_shape:
            description_vec = description_vec[:, :expected_tfidf_shape]

        # Combine numeric features and scale them
        numeric_features = np.array([[issue.severity, issue.occurrence, issue.detection]])
        numeric_features_scaled = scaler.transform(numeric_features)

        # Main model (neural network) prediction
        neural_net_prediction = model.predict([description_vec, numeric_features_scaled])

        if not neural_net_prediction or np.isnan(neural_net_prediction[0][0]):
            logger.warning("Model returned an invalid or empty prediction")
            raise ValueError("Model prediction invalid.")

        predicted_time_nn = float(neural_net_prediction[0][0])

        # Log the input and output of the prediction
        logger.info(f"Input: {issue.dict()}, Neural Net Prediction: {predicted_time_nn:.2f}")

        # Use neural network prediction directly
        final_predicted_time = predicted_time_nn

        # Apply a default frequency weight since issue_frequency is removed
        severity_weight = 1.2 if issue.severity >= 6 else 1
        frequency_weight = 1.1 if issue.occurrence >= 5 else 1
        weighted_time = final_predicted_time * severity_weight * frequency_weight

        # Generate recommendation
        recommendation = generate_recommendation(issue, final_predicted_time, rpn)

        # Confidence interval for predicted time
        lower_bound = max(0, final_predicted_time - 0.10 * final_predicted_time)
        upper_bound = final_predicted_time + 0.10 * final_predicted_time

        # Format time (automatically convert between minutes and hours)
        formatted_predicted_time = format_time(final_predicted_time)
        formatted_weighted_time = format_time(weighted_time)
        formatted_lower_bound = format_time(lower_bound)
        formatted_upper_bound = format_time(upper_bound)

        logger.info(f"Prediction successful: {formatted_predicted_time}")

        # Explicitly returning a JSONResponse with status code 200
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "Prediction successful",
                "predicted_time": formatted_predicted_time,
                "lower_bound_time": formatted_lower_bound,
                "upper_bound_time": formatted_upper_bound,
                "weighted_time": formatted_weighted_time,
                "frequency_weight": frequency_weight,
                "rpn": rpn,
                "recommended_solution": recommendation
            }
        )
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
