import numpy as np
import pandas as pd
import re
from keras.models import Model
from keras.layers import Input, Dense, concatenate, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import shap
import nltk

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the solutions dictionary
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
    text = re.sub(r'\W', ' ', text)  # Replace non-word characters with space
    text = text.lower()  # Convert to lowercase
    tokens = [word for word in text.split() if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Expanded dataset with more variations
new_data = {
    "description": [
        "Failure in motor", "Pump leakage", "Equipment overheating", 
        "Sensor malfunction", "Electrical short", "Fan failure", 
        "Valve not closing", "Leak detected", "Overheating issue", 
        "Motor malfunction", "Pump not working", "Sensor reading error",
        "Bearing failure", "Lubrication issue", "Motor coil short", 
        "Rotor imbalance", "Impeller damage", "Cavitation in pump",
        "Thermal expansion", "Cooling fan obstruction", "Hydraulic fluid leak", 
        "Controller malfunction", "Temperature sensor failure", "Heat exchanger clogged"
    ],
    "severity": [8, 5, 7, 4, 9, 6, 3, 7, 8, 5, 6, 4, 9, 8, 6, 7, 5, 9, 6, 8, 5, 7, 4, 6],
    "occurrence": [6, 7, 8, 3, 9, 5, 6, 8, 7, 4, 5, 6, 8, 9, 7, 8, 6, 9, 5, 6, 8, 9, 7, 6],
    "detection": [5, 4, 6, 7, 3, 8, 4, 5, 6, 5, 7, 6, 4, 5, 7, 6, 5, 4, 7, 6, 5, 4, 6, 7],
    "total_downtime": [120.0, 45.0, 80.0, 30.0, 90.0, 60.0, 15.0, 40.0, 75.0, 50.0, 20.0, 10.0, 140.0, 60.0, 110.0, 80.0, 100.0, 85.0, 60.0, 50.0, 75.0, 90.0, 45.0, 80.0],
    "timeframe_to_fix": [3, 1, 2, 1, 3, 2, 1, 2, 3, 1, 2, 1, 3, 2, 3, 2, 3, 3, 2, 2, 2, 3, 2, 1]
}

# Calculate RPN based on severity, occurrence, and detection
df = pd.DataFrame(new_data)
df['RPN'] = df['severity'] * df['occurrence'] * df['detection'] / 1000  # Normalized RPN for scale

# Preprocess text data
df['description'] = df['description'].apply(preprocess_text)

# TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=20)
X_text = tfidf.fit_transform(df['description'].values).toarray()

# Combine numeric features for modeling
numeric_features = df[['severity', 'occurrence', 'detection', 'total_downtime']].values
scaler = StandardScaler()
numeric_features_scaled = scaler.fit_transform(numeric_features)

# Define target (RPN)
rpn = df['RPN'].values

# Train-Test Split
X_train_text, X_test_text, X_train_num, X_test_num, y_train, y_test = train_test_split(
    X_text, numeric_features_scaled, rpn, test_size=0.3, random_state=42
)

# Model definition with advanced architecture
text_input = Input(shape=(X_train_text.shape[1],), name='text_input')
numeric_input = Input(shape=(4,), name='numeric_input')  # Four numeric features

dense_text = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(text_input)
dropout_text = Dropout(0.2)(dense_text)

concat_layer = concatenate([dropout_text, numeric_input])

# Add further dense layers with batch normalization
dense_1 = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(concat_layer)
batch_norm_1 = BatchNormalization()(dense_1)
dropout_1 = Dropout(0.3)(batch_norm_1)

dense_2 = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(dropout_1)
batch_norm_2 = BatchNormalization()(dense_2)
dropout_2 = Dropout(0.3)(batch_norm_2)

output = Dense(1, activation='linear', name='rpn_output')(dropout_2)

model = Model(inputs=[text_input, numeric_input], outputs=output)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')

# Checkpoint, early stopping, and learning rate reduction on plateau
checkpoint = ModelCheckpoint('best_rpn_model_advanced.keras', monitor='val_loss', save_best_only=True, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001, verbose=1)

# Model training
history = model.fit([X_train_text, X_train_num], y_train, epochs=150, batch_size=16,
                    validation_data=([X_test_text, X_test_num], y_test),
                    callbacks=[checkpoint, early_stopping, reduce_lr])

# Plotting training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('training_validation_loss_advanced.png')
plt.show()

# Model evaluation
try:
    y_pred = model.predict([X_test_text, X_test_num])
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger.info(f'Model Performance:\n RMSE: {rmse}\n MAE: {mae} \n R² Score: {r2}')
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
numeric_feature_names = ['severity', 'occurrence', 'detection', 'total_downtime']
feature_names = text_feature_names + numeric_feature_names  # Combine feature names

# Sort SHAP values by their absolute mean importance
sort_inds = np.argsort(np.abs(shap_values).mean(0))

# Generate summary plot of SHAP values
shap.summary_plot(
    shap_values, 
    features=X_test_combined, 
    feature_names=np.array(feature_names),
    show=False  # Prevent the figure from diplaying
)

# Save the trained model and necessary scalers for future use
model.save('rpn_predictor_model.keras')
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

logger.info("Model and scalers saved successfully.")

