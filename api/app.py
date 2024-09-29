from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
import numpy as np
import joblib
import logging
from fastapi import Response
from typing import List
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app with basic metadata
app = FastAPI(
    title="Maintenance Issue Prediction API",
    description="Predicts time to resolve maintenance issues based on description, severity, and other features.",
    version="1.1.0"
)

# Setup CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust origins as per your frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logging with timestamps
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic model for request validation
class MaintenanceIssue(BaseModel):
    description: str
    severity: float = Field(..., gt=0, lt=11, description="Severity should be between 1 and 10")
    total_downtime: float = Field(..., gt=0, description="Total downtime must be positive")
    oee: float = Field(..., gt=0, lt=1.1, description="OEE should be between 0 and 1")
    issue_frequency: int = Field(..., ge=0, description="Issue frequency must be non-negative")

# Load model and other assets once using FastAPI dependency injection
def get_model_assets():
    try:
        model = joblib.load("maintenance_issue_model.joblib")
        scaler = joblib.load("scaler.joblib")
        tfidf = joblib.load("tfidf_vectorizer.joblib")
        return model, scaler, tfidf
    except Exception as e:
        logger.error(f"Failed to load model or assets: {e}")
        raise HTTPException(status_code=500, detail="Model loading failed")

# Root endpoint for testing API status
@app.get("/")
async def read_root():
    return {"message": "API is up and running!"}

# Health Check to ensure all components are loaded correctly
@app.get("/health/")
async def health_check():
    try:
        model, scaler, tfidf = get_model_assets()
        return {"status": "healthy"}
    except Exception:
        raise HTTPException(status_code=500, detail="API health check failed")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

# Predicting maintenance issue (single issue)
@app.post("/predict/")
async def predict_issue(issue: MaintenanceIssue, assets=Depends(get_model_assets)):
    model, scaler, tfidf = assets
    logger.info(f"Received issue description: {issue.description}")

    try:
        # TF-IDF transformation
        description_vec = tfidf.transform([issue.description]).toarray()

        # Scale numeric features
        numeric_features = np.array([[issue.severity, issue.total_downtime, issue.oee]])
        numeric_features_scaled = scaler.transform(numeric_features)

        # Combine inputs into a single array for the model
        combined_input = np.hstack((description_vec, numeric_features_scaled))

        # Predict timeframe
        prediction = model.predict(combined_input)

        # Extract the first value from the prediction (assuming it's an array)
        predicted_time = float(prediction[0])

        # Apply frequency-based weight to the prediction
        frequency_weight = 1 + (issue.issue_frequency / 10)
        weighted_time = predicted_time * frequency_weight

        # Generate recommended solution based on predicted time
        recommended_solution = f"Based on the predicted time of {predicted_time:.2f} hours, consider allocating resources for efficient resolution."

        logger.info(f"Prediction successful: Predicted time: {predicted_time}, Weighted time: {weighted_time}")

        return {
            "predicted_time": predicted_time,
            "weighted_time": weighted_time,
            "frequency_weight": frequency_weight,
            "recommended_solution": recommended_solution
        }
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed, please check input data")

# Batch prediction for handling multiple issues at once
@app.post("/predict_batch/")
async def predict_batch(issues: List[MaintenanceIssue], assets=Depends(get_model_assets)):
    model, scaler, tfidf = assets
    predictions = []

    for issue in issues:
        try:
            # TF-IDF transformation
            description_vec = tfidf.transform([issue.description]).toarray()

            # Scale numeric features
            numeric_features = np.array([[issue.severity, issue.total_downtime, issue.oee]])
            numeric_features_scaled = scaler.transform(numeric_features)

            # Combine inputs into a single array for the model
            combined_input = np.hstack((description_vec, numeric_features_scaled))

            # Predict timeframe
            prediction = model.predict(combined_input)

            # Extract the first value from the prediction
            predicted_time = float(prediction[0])

            # Apply frequency-based weight
            frequency_weight = 1 + (issue.issue_frequency / 10)
            weighted_time = predicted_time * frequency_weight

            # Generate recommended solution
            recommended_solution = f"Allocate resources for a predicted time of {predicted_time:.2f} hours."

            predictions.append({
                "description": issue.description,
                "predicted_time": predicted_time,
                "weighted_time": weighted_time,
                "frequency_weight": frequency_weight,
                "recommended_solution": recommended_solution
            })
        except Exception as e:
            logger.error(f"Error during batch prediction for issue {issue.description}: {e}")
            predictions.append({
                "description": issue.description,
                "error": "Prediction failed, check input data"
            })

    return {"predictions": predictions}

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
