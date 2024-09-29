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

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize the logger for better error handling
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample Data Schema
data = {
    "description": ["Failure in motor", "Pump leakage", "Equipment overheating", "Sensor malfunction", "Electrical short"],
    "severity": [8, 5, 7, 4, 9],
    "total_downtime": [120, 45, 80, 30, 90],
    "oee": [0.8, 0.9, 0.7, 0.85, 0.6],
    "failure_modes": ["Motor failure", "Leakage", "Overheating", "Sensor issue", "Short circuit"],
    "timeframe_to_fix": [3, 1, 2, 1, 3]
}

# Convert to DataFrame
df = pd.DataFrame(data)

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
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove non-alphanumeric characters
    text = text.lower()  # Lowercase the text
    tokens = [word for word in text.split() if word not in stop_words]  # Remove stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize tokens
    return ' '.join(tokens)

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

# Train the model (epochs increased to 50)
logger.info("Starting model training...")
model.fit([X_train_text, X_train_num], y_train, epochs=50, batch_size=32, validation_data=([X_test_text, X_test_num], y_test))

# Model Evaluation
y_pred = model.predict([X_test_text, X_test_num])
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
logger.info(f'RMSE: {rmse:.2f}')

# Save model and scalers
model.save('issue_predictor_model.keras')  # Save the Keras model
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

# Load pre-trained model and scalers
def load_pretrained_assets():
    try:
        model = load_model('issue_predictor_model.keras')
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('tfidf.pkl', 'rb') as f:
            tfidf = pickle.load(f)
        return model, scaler, tfidf
    except Exception as e:
        logger.error(f"Error loading pretrained assets: {e}")
        return None, None, None

model, scaler, tfidf = load_pretrained_assets()

# Recommendation System using pre-trained model
def recommend_solution(description, tfidf_model, numeric_data, issue_frequency):
    try:
        # Convert the new description to TF-IDF format
        description_vec = tfidf_model.transform([description]).toarray()

        # Predict timeframe using the pre-trained model
        predicted_time = model.predict([description_vec, numeric_data])

        # Simple recommendation logic
        if issue_frequency > 5:
            recommended_solution = "This issue occurs frequently. Consider preventive maintenance or upgrading equipment."
        else:
            recommended_solution = "The issue is rare. Proceed with standard troubleshooting procedures."

        # Weighted time based on frequency
        frequency_weight = 1 + (issue_frequency / 10)
        weighted_time = predicted_time[0][0] * frequency_weight

        return recommended_solution, weighted_time, frequency_weight
    except Exception as e:
        logger.error(f"Error during recommendation: {e}")
        return "Recommendation error", None, None

# Test the pre-trained model
def test_model(issue_desc, severity, downtime, oee, issue_frequency):
    try:
        # Convert the issue description to TF-IDF format
        issue_vec = tfidf.transform([issue_desc]).toarray()

        # Scale the numeric features
        numeric_features = np.array([[severity, downtime, oee]])
        numeric_features_scaled = scaler.transform(numeric_features)

        # Get the recommended solution and weighted time
        recommended_solution, predicted_timeframe, frequency_weight = recommend_solution(issue_desc, tfidf, numeric_features_scaled, issue_frequency)

        # Print out the results
        logger.info(f"Issue Description: {issue_desc}")
        logger.info(f"Recommended Solution: {recommended_solution}")
        logger.info(f"Predicted Time to Fix: {predicted_timeframe:.2f} hours")
        logger.info(f"Frequency Weight Applied: {frequency_weight:.2f}")
    except Exception as e:
        logger.error(f"Error during model testing: {e}")

# Example Test Case with pre-trained model
test_model("Pump malfunction", 7, 85, 0.75, issue_frequency=6)
