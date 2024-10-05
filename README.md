
 CMOOS MODEL for prediction maintanence

# Table of Contents

1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Project Files](#project-files)
4. [Libraries Used](#libraries-used)
5. [Project Structure](#project-structure)
6. [Usage Instructions](#usage-instructions)
7. [Example Dataset](#example-dataset)
8. [Features Description](#features-description)
9. [Performance Metrics](#performance-metrics)
10. [API Overview](#api-overview)
11. [API Installation and Usage](#api-installation-and-usage)
12. [Environment Variables](#environment-variables)
13. [Dashboard Features](#dashboard-features)
14. [Running the Dashboard](#running-the-dashboard)
15. [Key Code Components](#key-code-components)
16. [Logging](#logging)
17. [Future Improvements](#future-improvements)
18. [Conclusion](#conclusion)


# Project Overview

Project Description

- The CMOOS model is developed to predict the Risk Priority Number (RPN) for equipment-related issues by combining both deep learning and machine learning techniques. It processes two types of data: textual descriptions of equipment problems and numeric features such as severity, occurrence, and detection. Through the integration of natural language processing (NLP) and numeric data fusion, the model aims to significantly enhance the accuracy of equipment failure predictions, helping to improve maintenance strategies.

Key components of the project include the use of pre-trained GloVe embeddings to capture the semantic meaning of the text data and the application of a hybrid model structure. This structure consists of a GRU-based neural network for processing text inputs and fully connected layers for numeric data, making the model capable of predicting both the likelihood and the severity of equipment failures in various industrial environments.


**Goal**

The overarching goal of this project is to build an intelligent prediction system that:

- Accurately predicts Risk Priority Numbers (RPN): By fusing textual issue descriptions with numeric data, the model can generate reliable RPN scores to assess the risk associated with potential equipment failures.Enhances maintenance strategies: By providing accurate predictions, the model can help maintenance teams prioritize tasks more effectively.

- Utilizes deep learning for textual data: Incorporating pre-trained embeddings and a GRU network to handle text inputs, ensuring the model extracts meaningful information from equipment failure descriptions.

- Optimizes model performance: Leveraging advanced techniques such as data augmentation for text inputs, and implementing a dynamic learning rate scheduler to ensure efficient training and strong generalization on new, unseen data.

Ultimately, the CMOOS model aims to be a robust, predictive tool for industrial maintenance applications, helping companies anticipate and address equipment failures before they escalate into costly issues.


# Project Files

- *[cmoos_model.py](cmoos_model.py)*: Contains the code to preprocess data, build the model, and train prediction model.

- *glove.6B.100d.txt*: Pre-trained GloVe embeddings used for text input.

**NB: The file will automatically download when running the model**

- *[issue_predictor_model.keras](issue_predictor_model.keras)*: The saved best model during training.

- [training_validation_loss_advanced.png](training_validation_loss_advanced.png)*: Visualizes the model loss across epochs.


# Libraries Used
- **TensorFlow / Keras**: For building the neural network model.

- **scikit-learn**: For machine learning utilities like data splitting, scaling, and model evaluation.

- **nltk**: For natural language preprocessing, including lemmatization and stopword removal.

- **GloVe Embeddings**: To capture word semantics.

- **gdown**: For downloading the GloVe embeddings.

- **Matplotlib**: For plotting and visualizing the model training process.

- **Logging**: To track and monitor model training.
# Project Structure

1. **Data Preprocessing**:

- Text descriptions are preprocessed to remove stop words, lemmatize tokens, and correct spelling.
- A synonym replacement method is applied to augment text data, enhancing model robustness.

2. **TF-IDF Vectorization**:

- Text data is vectorized using TF-IDF and combined with numeric features for comprehensive input.

3. **Model Architecture**

- A Bidirectional GRU layer processes text inputs, while numeric features pass through dense layers.

- Batch Normalization and Dropout layers prevent overfitting.

- Outputs a predicted RPN value based on combined text and numeric data.

4. # Model Training:

- The model is trained on the processed data using Mean Squared Error (MSE) as the loss function and Adam optimizer.

- The learning rate is adjusted dynamically using a learning rate scheduler.

- Best model checkpoints are saved based on validation loss.


# Usage Instructions 

1. Set Up Environment

*install the required Python packages*

```bash
pip install -r requirements.txt
``` 

2. Download GloVe Embeddings
The project uses pre-trained GloVe embeddings. If not present, the embeddings will be automatically downloaded by the script.

3. Train the Model.

*Run the cmoos script to train the 
prediction model*:

python cmoos_model.py

The model will train on the text descriptions and numeric data to predict RPN values, and the results will be saved as checkpoints.

4. Visualize Model Training.
*After training, a plot of the training vs validation loss will be generated*:

![training_validation_loss_advanced.png](training_validation_loss_advanced.png)

5. Evaluate the Model
*Once trained, the model will predict RPN values on unseen test data. The performance metrics (RMSE, MAE) will be logged in the console*.


Example Dataset

| Description            | Severity | Occurrence | Detection | RPN   |
|------------------------|----------|------------|-----------|-------|
| Motor failure           | 8        | 6          | 5         | 0.24  |
| Pump leakage            | 5        | 7          | 4         | 0.14  |
| Equipment overheating   | 7        | 8          | 6         | 0.34  |
| Sensor malfunction      | 4        | 3          | 7         | 0.084 |


**Note: This dataset is augmented using synonym replacement to create diverse variations.**

 *Features Description*

- **Description**: Textual description of the failure or issue.

- **Severity**: Numeric rating of the severity of the failure (1-10).
- **Occurrence**: Numeric rating indicating the frequency of the failure (1-10).

- **Detection**: Numeric rating representing how likely it is to detect the failure before it happens (1-10).

- **RPN (Target)**: Risk Priority Number, calculated as:
  \[
  \text{RPN} = \frac{\text{Severity} \times \text{Occurrence} \times \text{Detection}}{1000}
  \]


# Performance Metrics

*After training, the following metrics are logged*:

- Training RMSE: Measures the root mean squared error on training data.

- Test RMSE: Evaluates model performance on unseen test data.

- Training MAE: Measures mean absolute error during training.

- Test MAE: Quantifies prediction error on test data.


# API Overview

This project also includes an API built using FastAPI to predict the Risk Priority Number (RPN) and provide maintenance issue resolution times based on textual descriptions and numeric features like severity, occurrence, and detection

**Key Features of the API**:

- Predict RPN: Combines numeric features and text description to predict the RPN of an issue.
Predict Time to Resolve: Uses a trained neural network to predict the time it will take to resolve an issue.

- Generate Recommendations: Based on the severity, RPN, and predicted time, the API offers tailored maintenance recommendations.

- CORS Support: Configured to allow cross-origin requests for ease of integration during development.

**API Endpoints**

1. Root Endpoint.

GET /

*Description: Welcome message to confirm the API is running*.

Response

```json
{
  "message": "Welcome to the Maintenance Issue Prediction API!"
}
```

2. Predict Issue Resolution Time

POST /predict/

*Description: Predicts the maintenance issue resolution time and calculates RPN based on the input.*

Request Body:

```json
{
  "description": "Motor failure",
  "severity": 8,
  "occurrence": 6,
  "detection": 5
}
```
- ```description```: Textual  description of the issue.

- ```severity```: Numeric rating for the severity (1-10 scale).

- ```occurrence```: Numeric rating indicating how often the issue occurs (1-10 scale).

- ```detection```: Numeric rating for how easily the issue can be detected (1-10 scale).

*Response*

```json
{
  "message": "Prediction successful",
  "predicted_time": 7.8,
  "lower_bound_time": 7.02,
  "upper_bound_time": 8.58,
  "weighted_time": 9.36,
  "frequency_weight": 1.2,
  "rpn": 240,
  "recommended_solution": "High severity issue; urgent response required. Allocate extra resources for longer repair times."
}
```
- ```predicted_time```: Predicted hours to resolve the issue.

- ```lower_bound_time```: Lower bound of the confidence interval for the predicted time.

- ```upper_bound_time```: Upper bound of the confidence interval for the predicted time.

- ```weighted_time```: Adjusted time based on frequency of occurrence.
- ```rpn```: Calculated Risk Priority Number.

- ```recommended_solution```: Suggested action based on severity, RPN, and estimated repair time.

# API Installation and Usage

1. Clone the Repository

```bash 
git clone https://github.com/mosixty1234/cmoos_model-api.git
```

2. Install Requirements
**Ensure you have Python 3.8+ installed. Install dependencies using pip:**

```bash 
pip install -r requirements.txt
```
3. Start the api

*You can run the FastAPI application using Uvicorn:*

```bash
python3 -m uvicorn app:app --reload
```

4. Access the API Documentation

*After starting the server, you can explore the API and try requests directly via the interactive documentation at:*

- Swagger UI:http://localhost:8000/docs

# Environment Variables

*The application relies on several environment variables to load the necessary assets. These can be set in a .env file*:

- ```MODEL_PATH```: Path to the saved neural network model (default: issue_predictor_model.keras).

- ```SCALER_PATH ```: Path to the scaler file for numeric features (default: scaler.pkl).

- ```TFIDF_PATH```: Path to the TF-IDF vectorizer for text features(default: tfidf.pkl)..


# Dashboard Overview

This project includes a Dashboard built with Plotly Dash to interact with the Maintenance Issue Prediction API. The dashboard allows users to enter details about a maintenance issue, get predictions on the time needed to resolve the issue, and receive recommendations based on the predicted Risk Priority Number (RPN).

# Dashboard Features

- Input Fields:

  - Issue Description: Textual description of the maintenance issue.

   - Severity, Occurrence,   Detection: Numeric inputs on a scale from 1-10 to represent the severity, likelihood of occurrence, and detection capability.

  - Total Downtime: Numeric input representing the estimated downtime in hours.


- RPN Calculation:

  - Automatically calculates and displays the Risk Priority Number (RPN) based on the severity, occurrence, and detection values entered

- Prediction Output:

  - Displays the predicted time to resolve the issue.

  - Provides a weighted estimate based on occurrence.

  - Offers a recommendation to address the issue.

- Issue Frequency Chart:

  - Visualizes the frequency of common maintenance issues using a bar chart.


# Running the Dashboard

*Prerequisites: Ensure the following Python packages are installed:*

```bash
pip install dash plotly pandas requests dash-bootstrap-components
```
# Steps to Run

1. Start the FastAPI Prediction API
*Ensure the FastAPI prediction API is running locally (or replace the API_URL in the code if hosted elsewhere). Use the following command to start the API server:*

```bash
python3 -m uvicorn app:app --reload
```

2. Run the Dash Application
*Execute the dashboard application using the command*:

```bash
python3 dashboard.py
```

3. Open the Dashboard.

**Open your web browser and navigate to http://127.0.0.1:8050 to access the dashboard.**


# Key Code Components

*The dashboard integrates the following key components*: 

1. API Integration: Connects to the FastAPI endpoint to submit issue details and retrieve predictions.

2. Interactive Input Form: Collects user inputs for maintenance issue details and calculates the RPN dynamically.

3. Issue Frequency Chart: Displays a bar chart using Plotly to represent the frequency of common issues.

# Logging

The dashboard logs key actions and errors in the dash_app.log file. Check this file for troubleshooting information if any issues arise while running the application.

# Future Improvements

- More Robust Augmentation: Explore advanced text augmentation techniques like back-translation.

- Hyperparameter Tuning: Use techniques like grid search to fine-tune model hyperparameters for better results.

- Explainability: Incorporate SHAP to interpret model predictions.

# Conclusion

This project showcases a hybrid approach combining deep learning for text data and machine learning for numeric features to predict the Risk Priority Number. The model's ability to generalize from diverse input sources makes it highly adaptable to various industrial applications.
