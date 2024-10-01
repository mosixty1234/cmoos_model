from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import pickle
import logging
from tensorflow.keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI(
    title="Maintenance Issue Prediction API",
    description="Predicts time to resolve maintenance issues based on description, severity, and other features.",
    version="2.0.0"
)

# Load environment variables (paths)
MODEL_PATH = 'issue_predictor_model.keras'
SCALER_PATH = 'scaler.pkl'
TFIDF_PATH = 'tfidf.pkl'

# Setup CORS middleware for development
origins = ["http://127.0.0.1:8080"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["Authorization", "Content-Type"],
)

# Setup logging
logging.basicConfig(
    filename="app.log",
    filemode="a",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load model and preprocessing assets on startup
@app.on_event("startup")
async def startup_event():
    global model, scaler, tfidf
    try:
        model, scaler, tfidf = load_assets()
        logger.info("Model, scaler, and TFIDF loaded successfully.")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise HTTPException(status_code=500, detail="Failed to load assets.")

# Load model and assets
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

# Health check endpoint
@app.get("/health/")
async def health_check():
    try:
        model, scaler, tfidf = load_assets()
        return {"status": "healthy"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="API health check failed")

# Request validation using Pydantic
class MaintenanceIssue(BaseModel):
    description: str
    severity: float = Field(..., gt=0, lt=11, description="Severity must be between 1 and 10")
    total_downtime: float = Field(..., gt=0, description="Total downtime must be positive")
    rpn: float = Field(..., gt=0, lt=101, description="RPN must be between 0 and 100")
    issue_frequency: int = Field(..., ge=0, description="Issue frequency must be non-negative")

# Predict maintenance issue resolution time
@app.post("/predict/")
async def predict_issue(issue: MaintenanceIssue):
    try:
        logger.info(f"Processing issue: {issue.description}")

        # Transform input description using the TFIDF vectorizer
        description_vec = tfidf.transform([issue.description]).toarray()

        # Combine numeric features and scale them
        numeric_features = np.array([[issue.severity, issue.total_downtime, issue.rpn]])
        numeric_features_scaled = scaler.transform(numeric_features)

        # Model prediction
        prediction = model.predict([description_vec, numeric_features_scaled])

        if prediction is None or len(prediction) == 0:
            raise ValueError("Model returned an empty prediction")

        predicted_time = float(prediction[0][0])

        # Log the input and output of the prediction
        logger.info(f"Input: {issue.dict()}, Predicted Time: {predicted_time:.2f}")

        # Apply weighting based on issue frequency
        frequency_weight = 1 + (issue.issue_frequency / 10)
        weighted_time = predicted_time * frequency_weight

        # Generate enhanced recommendation
        recommendation = generate_recommendation(issue, predicted_time, weighted_time)

        logger.info(f"Prediction successful: {predicted_time} hours")

        return {
            "predicted_time": round(predicted_time, 2),
            "weighted_time": round(weighted_time, 2),
            "frequency_weight": frequency_weight,
            "recommended_solution": recommendation
        }
    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

# Improved recommendation based on severity, RPN, and downtime
def generate_recommendation(issue: MaintenanceIssue, predicted_time: float, weighted_time: float) -> str:
    if issue.severity >= 8:
        return (
            f"The issue has a high severity level ({issue.severity}/10), which demands immediate attention. "
            f"The predicted resolution time is approximately {predicted_time:.2f} hours, and considering past occurrences, "
            f"the weighted resolution time is {weighted_time:.2f} hours. It is recommended to assign experienced personnel to "
            "mitigate the potential downtime."
        )
    elif issue.rpn > 70:
        return (
            f"High RPN detected ({issue.rpn}/100), which indicates a critical risk. "
            f"The issue should be resolved within {predicted_time:.2f} hours. Ensure preventive measures are in place "
            "to avoid recurrence."
        )
    elif issue.total_downtime > 5:
        return (
            f"The system has already experienced significant downtime ({issue.total_downtime:.2f} hours). "
            f"The predicted resolution time is {predicted_time:.2f} hours. To minimize further impact, "
            "ensure sufficient resource allocation and monitor for potential cascading failures."
        )
    else:
        return (
            f"The issue is expected to be resolved within {predicted_time:.2f} hours. "
            f"Considering previous issue frequency, the weighted resolution time is {weighted_time:.2f} hours. "
            "Allocate resources efficiently and ensure prompt follow-up after resolution."
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
