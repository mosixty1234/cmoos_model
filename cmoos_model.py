
import numpy as np
import pandas as pd
import re  # Import for regular expressions
from keras.models import Model
from keras.layers import Input, Dense, concatenate, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.exceptions import UndefinedMetricWarning
import warnings
import shap

# Ignore warnings for undefined metrics
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Set up logging for tracking and debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define solutions dictionary for recommendations
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

# Preprocess text function
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r'\W', ' ', text)  # Replace non-word characters with a space
    text = text.lower()  # Convert to lowercase
    tokens = [word for word in text.split() if word not in stop_words]  # Remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize each word
    return ' '.join(tokens)

# Recommend solution based on description
def recommend_solution(description):
    description = preprocess_text(description)
    for keyword, solution in solutions.items():
        if keyword in description:
            return solution
    return "No specific solution found. Please investigate further."

# Validate data function
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
    if not np.issubdtype(df['RPN'].dtype, np.floating):
        logger.error("RPN must be a float.")
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
    if df['RPN'].min() < 0 or df['RPN'].max() > 100:
        logger.error("RPN must be between 0 and 100.")
        return False
    return True

# Sample data with more entries to ensure enough test samples
new_data = {
    "description": [
        "Failure in motor", "Pump leakage", "Equipment overheating", 
        "Sensor malfunction", "Electrical short", "Fan failure", 
        "Valve not closing", "Leak detected", "Overheating issue", 
        "Motor malfunction", "Pump not working", "Sensor reading error"
    ],
    "severity": [8, 5, 7, 4, 9, 6, 3, 7, 8, 5, 6, 4],
    "total_downtime": [120.0, 45.0, 80.0, 30.0, 90.0, 60.0, 15.0, 40.0, 75.0, 50.0, 20.0, 10.0],
    "RPN": [0.8, 0.9, 0.7, 0.85, 0.6, 0.75, 0.65, 0.9, 0.55, 0.5, 0.6, 0.4],
    "timeframe_to_fix": [3, 1, 2, 1, 3, 2, 1, 2, 3, 1, 2, 1]
}

# Create DataFrame and preprocess descriptions
df = pd.DataFrame(new_data)
df['description'] = df['description'].astype(str)

# Validate input data
if not validate_data(df):
    raise ValueError("Input data validation failed. Check logs for more information.")

# Fill missing descriptions
df.fillna({'description': 'No description provided'}, inplace=True)
df['severity'] = df['severity'].astype(int)
df['total_downtime'] = df['total_downtime'].astype(float)
df['RPN'] = df['RPN'].astype(float)
df['timeframe_to_fix'] = df['timeframe_to_fix'].astype(int)

# Text preprocessing
df['description'] = df['description'].apply(preprocess_text)

# TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=11)
X_text = tfidf.fit_transform(df['description'].values).toarray()

# Numeric feature scaling
numeric_features = df[['severity', 'total_downtime', 'RPN']].values
scaler = StandardScaler()
numeric_features_scaled = scaler.fit_transform(numeric_features)

# Define target variable
timeframe = df['timeframe_to_fix'].values

# Train-Test Split with a larger test size to ensure enough samples
X_train_text, X_test_text, X_train_num, X_test_num, y_train, y_test = train_test_split(
    X_text, numeric_features_scaled, timeframe, test_size=0.3, random_state=42
)

# Model definition
text_input = Input(shape=(X_train_text.shape[1],), name='text_input')
numeric_input = Input(shape=(3,), name='numeric_input')

dense_text = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(text_input)
dropout_text = Dropout(0.2)(dense_text)
concat_layer = concatenate([dropout_text, numeric_input])

dense_1 = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(concat_layer)
dropout_1 = Dropout(0.3)(dense_1)

dense_2 = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(dropout_1)
dropout_2 = Dropout(0.3)(dense_2)

output = Dense(1, activation='linear', name='timeframe_output')(dropout_2)

model = Model(inputs=[text_input, numeric_input], outputs=output)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Checkpoint and early stopping
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Model training
try:
    logger.info("Starting model training...")
    model.fit([X_train_text, X_train_num], y_train, epochs=50, batch_size=32,
              validation_data=([X_test_text, X_test_num], y_test),
              callbacks=[checkpoint, early_stopping])
except Exception as e:
    logger.error(f"An error occurred during model training: {e}")
    raise

# Model evaluation
try:
    y_pred = model.predict([X_test_text, X_test_num])
    
    if len(y_test) < 2:
        logger.warning("Not enough samples in test set to calculate R² score.")
    else:
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        logger.info(f'Model Performance:\n RMSE: {rmse}\n MAE: {mae}\n R² Score: {r2}')
except Exception as e:
    logger.error(f"Error during model evaluation: {e}")
    raise
   
# SHAP Explanation: Initialize SHAP for model interpretation
logger.info("Starting SHAP explanation...")
X_train_combined = np.concatenate((X_train_text, X_train_num), axis=1)  # Combine features for SHAP

# Use a lambda function for the model prediction in SHAP
explainer = shap.KernelExplainer(
    lambda x: model.predict([x[:, :X_train_text.shape[1]], x[:, X_train_text.shape[1]:]]), 
    X_train_combined
)

X_test_combined = np.concatenate((X_test_text, X_test_num), axis=1)  # Combine test features for SHAP
shap_values = explainer.shap_values(X_test_combined)  # Calculate SHAP values for the test set

# Check if SHAP values are in a list and handle accordingly
if isinstance(shap_values, list):
    shap_values = shap_values[0]

# Create a list of feature names for visualization
text_feature_names = [f'TF-IDF_{i}' for i in range(X_train_text.shape[1])]
numeric_feature_names = ['severity', 'total_downtime', 'oee']
feature_names = text_feature_names + numeric_feature_names  # Combine feature names

# Sort SHAP values by their absolute mean importance
sort_inds = np.argsort(np.abs(shap_values).mean(0))

# Generate summary plot of SHAP values
shap.summary_plot(
    shap_values, 
    features=X_test_combined, 
    feature_names=np.array(feature_names),
    show=False  # Prevent the figure from displaying
)
   
# Save the trained model and necessary scalers for future use
model.save('issue_predictor_model.keras')
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

logger.info("Model and scalers saved successfully.")

