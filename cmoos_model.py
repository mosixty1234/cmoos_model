import numpy as np
import pandas as pd
from keras.models import Model, load_model
from keras.layers import Input, Dense, concatenate
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import logging

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize the logger for better error handling
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Solutions based on issue description
solutions = {
    'motor': 'Replace motor bearings and test alignment.',
    'pump': 'Check pump seals and replace if necessary.',
    'overheat': 'Inspect cooling system, clean filters.',
    'sensor': 'Calibrate or replace sensor.',
    'short': 'Inspect wiring and replace damaged components.'
}

# Function to preprocess text
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r'\W', ' ', text)  # Remove non-alphanumeric characters
    text = text.lower()  # Lowercase the text
    tokens = [word for word in text.split() if word not in stop_words]  # Remove stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize tokens
    return ' '.join(tokens)

# Function to recommend solutions based on issue description
def recommend_solution(description):
    description = preprocess_text(description)  # Preprocess the text
    for keyword, solution in solutions.items():
        if keyword in description:
            return solution
    return "No specific solution found. Please investigate further."

# Sample new data
new_data = {
    "description": ["Failure in motor", "Pump leakage", "Equipment overheating", "Sensor malfunction", "Electrical short"],
    "severity": [8, 5, 7, 4, 9],
    "total_downtime": [120, 45, 80, 30, 90],
    "oee": [0.8, 0.9, 0.7, 0.85, 0.6],
    "timeframe_to_fix": [3, 1, 2, 1, 3]
}

# Convert to DataFrame
df = pd.DataFrame(new_data)

# Data Cleaning
df.fillna({'description': 'No description provided'}, inplace=True)
df['severity'] = df['severity'].astype(int)
df['total_downtime'] = df['total_downtime'].astype(float)
df['oee'] = df['oee'].astype(float)
df['timeframe_to_fix'] = df['timeframe_to_fix'].astype(int)

# Detect and remove outliers using IQR
Q1 = df[['severity', 'total_downtime', 'oee']].quantile(0.25)
Q3 = df[['severity', 'total_downtime', 'oee']].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[['severity', 'total_downtime', 'oee']] < (Q1 - 1.5 * IQR)) | (df[['severity', 'total_downtime', 'oee']] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Text Preprocessing
df['description'] = df['description'].apply(preprocess_text)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=11)
X_text = tfidf.fit_transform(df['description'].values).toarray()

# Numeric features (severity, downtime, OEE)
numeric_features = df[['severity', 'total_downtime', 'oee']].values
scaler = StandardScaler()
numeric_features_scaled = scaler.fit_transform(numeric_features)

# Targets (timeframe to fix)
timeframe = df['timeframe_to_fix'].values

# Train-Test Split
X_train_text, X_test_text, X_train_num, X_test_num, y_train, y_test = train_test_split(
    X_text, numeric_features_scaled, timeframe, test_size=0.2, random_state=42)

# Model Building
text_input = Input(shape=(X_train_text.shape[1],), name='text_input')
numeric_input = Input(shape=(3,), name='numeric_input')

# Dense layers for text data
dense_text = Dense(128, activation='relu')(text_input)

# Concatenate text and numeric features
concat_layer = concatenate([dense_text, numeric_input])

# Dense layers after concatenation
dense_1 = Dense(128, activation='relu')(concat_layer)
dense_2 = Dense(64, activation='relu')(dense_1)

# Output layer (predicting timeframe)
output = Dense(1, activation='linear', name='timeframe_output')(dense_2)

# Define and compile the model
model = Model(inputs=[text_input, numeric_input], outputs=output)
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model with new data
logger.info("Starting model training...")
model.fit([X_train_text, X_train_num], y_train, epochs=25, batch_size=32, validation_data=([X_test_text, X_test_num], y_test))

# Model Evaluation
y_pred = model.predict([X_test_text, X_test_num])
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
logger.info(f'RMSE: {rmse:.2f}')

# Save model and scalers
model.save('issue_predictor_model_updated.keras')  # Save the Keras model
with open('scaler_updated.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('tfidf_updated.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

# Load pre-trained model and scalers (dynamic retraining)
def load_pretrained_assets():
    try:
        model = load_model('issue_predictor_model_updated.keras')
        with open('scaler_updated.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('tfidf_updated.pkl', 'rb') as f:
            tfidf = pickle.load(f)
        return model, scaler, tfidf
    except Exception as e:
        logger.error(f"Error loading pretrained assets: {e}")
        return None, None, None

model, scaler, tfidf = load_pretrained_assets()

# Real-time retraining function
def retrain_model(new_data_df):
    try:
        logger.info("Retraining model with new data...")

        # Preprocess new data
        new_data_df['description'] = new_data_df['description'].apply(preprocess_text)
        X_text_new = tfidf.transform(new_data_df['description'].values).toarray()
        numeric_features_new = new_data_df[['severity', 'total_downtime', 'oee']].values
        numeric_features_scaled_new = scaler.transform(numeric_features_new)
        y_new = new_data_df['timeframe_to_fix'].values

        # Fine-tune model with new data
        model.fit([X_text_new, numeric_features_scaled_new], y_new, epochs=50, batch_size=32)

        logger.info("Model retrained successfully.")
        model.save('issue_predictor_model_updated.keras')  # Save updated model
    except Exception as e:
        logger.error(f"Error retraining model: {e}")

# Example call to retrain with new data and recommend solutions
def retrain_and_recommend(new_data_df):
    retrain_model(new_data_df)
    new_data_df['recommended_solution'] = new_data_df['description'].apply(recommend_solution)
    return new_data_df

# Example retrain and recommend solutions
retrain_and_recommend(pd.DataFrame({
    "description": ["Fan failure", "Pipe burst", "Valve malfunction"],
    "severity": [6, 8, 7],
    "total_downtime": [60, 120, 90],
    "oee": [0.85, 0.75, 0.8],
    "timeframe_to_fix": [2, 3, 4]
}))
