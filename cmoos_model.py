import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, Dense, concatenate
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import logging
import shap
import matplotlib

# Set matplotlib to a non-interactive backend for rendering plots without display
matplotlib.use('Agg')

# Download required NLTK resources for text processing
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize logger for tracking and debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define a dictionary for recommended solutions based on keywords in the issue description
solutions = {
    'motor': 'Replace motor bearings and test alignment.',
    'pump': 'Check pump seals and replace if necessary.',
    'overheat': 'Inspect cooling system, clean filters.',
    'sensor': 'Calibrate or replace sensor.',
    'short': 'Inspect wiring and replace damaged components.',
    'fan': 'Inspect fan blades and replace if damaged.',
    'valve': 'Check valve seals and replace if needed.',
    'leak': 'Inspect connections and tighten fittings.'
}

# Function to preprocess the issue description text
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    tokens = [word for word in text.split() if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Function to recommend solutions based on preprocessed issue description
def recommend_solution(description):
    description = preprocess_text(description)
    for keyword, solution in solutions.items():
        if keyword in description:
            return solution
    return "No specific solution found. Please investigate further."

# Function to validate the input DataFrame
def validate_data(df):
    if df.isnull().values.any():
        logger.error("Input data contains missing values.")
        return False
    
    if not pd.api.types.is_string_dtype(df['description']):
        logger.error("Description column must be of type string.")
        return False
    
    if not np.issubdtype(df['severity'].dtype, np.integer):
        logger.error("Severity must be an integer.")
        return False
    if not np.issubdtype(df['total_downtime'].dtype, np.floating):
        logger.error("Total Downtime must be a float.")
        return False
    if not np.issubdtype(df['oee'].dtype, np.floating):
        logger.error("OEE must be a float.")
        return False
    if not np.issubdtype(df['timeframe_to_fix'].dtype, np.integer):
        logger.error("Timeframe to fix must be an integer.")
        return False
    
    if df['severity'].min() < 1 or df['severity'].max() > 10:
        logger.error("Severity must be between 1 and 10.")
        return False
    if df['total_downtime'].min() < 0:
        logger.error("Total Downtime must be non-negative.")
        return False
    if df['oee'].min() < 0 or df['oee'].max() > 1:
        logger.error("OEE must be between 0 and 1.")
        return False
    
    return True

# Sample new data for testing the model
new_data = {
    "description": ["Failure in motor", "Pump leakage", "Equipment overheating", "Sensor malfunction", "Electrical short"],
    "severity": [8, 5, 7, 4, 9],
    "total_downtime": [120.0, 45.0, 80.0, 30.0, 90.0],
    "oee": [0.8, 0.9, 0.7, 0.85, 0.6],
    "timeframe_to_fix": [3, 1, 2, 1, 3]
}

# Convert sample data into a DataFrame for processing
df = pd.DataFrame(new_data)
df['description'] = df['description'].astype(str)

# Validate input data against defined rules
if not validate_data(df):
    raise ValueError("Input data validation failed. Check logs for more information.")

# Data Cleaning: Fill missing descriptions and ensure correct data types
df.fillna({'description': 'No description provided'}, inplace=True)
df['severity'] = df['severity'].astype(int)
df['total_downtime'] = df['total_downtime'].astype(float)
df['oee'] = df['oee'].astype(float)
df['timeframe_to_fix'] = df['timeframe_to_fix'].astype(int)

# Detect and remove outliers using Interquartile Range (IQR) method
Q1 = df[['severity', 'total_downtime', 'oee']].quantile(0.25)
Q3 = df[['severity', 'total_downtime', 'oee']].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[['severity', 'total_downtime', 'oee']] < (Q1 - 1.5 * IQR)) | (df[['severity', 'total_downtime', 'oee']] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Text Preprocessing: Apply the preprocessing function to the description column
df['description'] = df['description'].apply(preprocess_text)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=11)
X_text = tfidf.fit_transform(df['description'].values).toarray()

# Extract numeric features
numeric_features = df[['severity', 'total_downtime', 'oee']].values
scaler = StandardScaler()
numeric_features_scaled = scaler.fit_transform(numeric_features)

# Define target variable
timeframe = df['timeframe_to_fix'].values

# Train-Test Split
X_train_text, X_test_text, X_train_num, X_test_num, y_train, y_test = train_test_split(
    X_text, numeric_features_scaled, timeframe, test_size=0.2, random_state=42
)

# Model Building
text_input = Input(shape=(X_train_text.shape[1],), name='text_input')
numeric_input = Input(shape=(3,), name='numeric_input')

# Dense layers
dense_text = Dense(128, activation='relu')(text_input)
concat_layer = concatenate([dense_text, numeric_input])
dense_1 = Dense(128, activation='relu')(concat_layer)
dense_2 = Dense(64, activation='relu')(dense_1)
output = Dense(1, activation='linear', name='timeframe_output')(dense_2)

model = Model(inputs=[text_input, numeric_input], outputs=output)
model.compile(optimizer='adam', loss='mean_squared_error')

# Exception handling during model training
try:
    logger.info("Starting model training...")
    model.fit([X_train_text, X_train_num], y_train, epochs=50, batch_size=100, validation_data=([X_test_text, X_test_num], y_test))
except Exception as e:
    logger.error(f"An error occurred during model training: {e}")
    raise

# Model Evaluation: Predict and calculate performance metrics
try:
    y_pred = model.predict([X_test_text, X_test_num])
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger.info(f'Model Performance:\nRMSE: {rmse:.2f}\nMAE: {mae:.2f}\nR²: {r2:.2f}')
except Exception as e:
    logger.error(f"An error occurred during model evaluation: {e}")
    raise

# SHAP Explanation
try:
    logger.info("Starting SHAP explanation...")
    X_train_combined = np.concatenate((X_train_text, X_train_num), axis=1)
    explainer = shap.KernelExplainer(lambda x: model.predict([x[:, :X_train_text.shape[1]], x[:, X_train_text.shape[1]:]]), X_train_combined)
    X_test_combined = np.concatenate((X_test_text, X_test_num), axis=1)
    shap_values = explainer.shap_values(X_test_combined)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    text_feature_names = [f'TF-IDF_{i}' for i in range(X_train_text.shape[1])]
    numeric_feature_names = ['severity', 'total_downtime', 'oee']
    feature_names = text_feature_names + numeric_feature_names

    sort_inds = np.argsort(np.abs(shap_values).mean(0))

    shap.summary_plot(
        shap_values,
        features=X_test_combined,
        feature_names=np.array(feature_names),
        show=False
    )
except Exception as e:
    logger.error(f"An error occurred during SHAP explanation: {e}")
    raise

# Save model and scalers
try:
    with open("model.h5", "wb") as model_file:
        pickle.dump(model, model_file)
    with open("scaler.pkl", "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)
    with open("tfidf.pkl", "wb") as tfidf_file:
        pickle.dump(tfidf, tfidf_file)
    logger.info("Model, scaler, and TF-IDF vectorizer saved successfully.")
except Exception as e:
    logger.error(f"An error occurred while saving the model or preprocessors: {e}")
    raise
