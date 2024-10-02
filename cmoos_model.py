import numpy as np
import pandas as pd
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, GRU, Bidirectional, Dense, concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
import matplotlib.pyplot as plt
import shap
import random
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import nltk
import contractions
import pickle
from textblob import TextBlob

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Update solutions dictionary with additional issues and their corresponding solutions
solutions = {
   'motor': 'Replace motor bearings and test alignment.',
   'pump': 'Check pump seals and replace if necessary.',
   'overheat': 'Inspect cooling system, clean filters.',
   'sensor': 'Calibrate or replace sensor.',
   'short': 'Inspect wiring and replace damaged components.',
    'fan': 'Inspect fan blades and replace if damaged.',
    'valve': 'Check valve seals and replace if needed.',
    'leak': 'Inspect connections and tighten fittings.',
    'compressor': 'Inspect compressor coils and refrigerant levels.',
    'vibration': 'Check equipment balance and lubricate bearings.',
    'actuator': 'Replace worn-out actuator and reprogram controller.',
    'belt': 'Tighten or replace belt as necessary.',
   'bearing': 'Lubricate or replace worn bearings.',
    'circuit': 'Inspect circuit breakers and reset if necessary.',
    'filter': 'Clean or replace air filter to prevent clogging.',
   'fluid': 'Check fluid levels and top up if required.'
}

# Additional data entries for improving model robustness (ensure all lists are of the same length)
new_data = {
    "description": [
        "Failure in motor", "Pump leakage", "Equipment overheating", 
        "Sensor malfunction", "Electrical short", "Fan failure", 
        "Valve not closing", "Leak detected", "Overheating issue", 
        "Motor malfunction", "Pump not working", "Sensor reading error",
        "Bearing failure", "Lubrication issue", "Motor coil short", 
        "Rotor imbalance", "Impeller damage", "Cavitation in pump",
        "Thermal expansion", "Cooling fan obstruction", "Hydraulic fluid leak", 
        "Controller malfunction", "Temperature sensor failure", "Heat exchanger clogged",
        "Compressor failure", "Vibration issue", "Actuator malfunction",
        "Belt slipping", "Bearing overheating", "Electrical circuit short", 
        "Air filter blockage", "Low fluid levels", "Sensor miscalibration",
        "Controller fault", "Pump impeller worn out", "Vibration due to misalignment"
    ],
    "severity": [8, 5, 7, 4, 9, 6, 3, 7, 8, 5, 6, 4, 9, 8, 6, 7, 5, 9, 6, 8, 5, 7, 4, 6, 9, 8, 6, 7, 4, 5, 9, 6, 7, 5, 9, 8],
    "occurrence": [6, 7, 8, 3, 9, 5, 6, 8, 7, 4, 5, 6, 8, 9, 7, 8, 6, 9, 5, 6, 8, 9, 7, 6, 7, 6, 8, 9, 7, 8, 5, 9, 7, 8, 6, 3],
    "detection": [5, 4, 6, 7, 3, 8, 4, 5, 6, 5, 7, 6, 4, 5, 7, 6, 5, 4, 7, 6, 5, 4, 6, 7, 5, 4, 6, 7, 5, 6, 7, 6, 5, 4, 7, 3],
    "total_downtime": [120.0, 45.0, 80.0, 30.0, 90.0, 60.0, 15.0, 40.0, 75.0, 50.0, 20.0, 10.0, 140.0, 60.0, 110.0, 80.0, 100.0, 85.0, 60.0, 50.0, 75.0, 90.0, 45.0, 80.0, 100.0, 85.0, 60.0, 30.0, 70.0, 90.0, 55.0, 110.0, 65.0, 70.0, 90.0, 65.0],
    "timeframe_to_fix": [3, 1, 2, 1, 3, 2, 1, 2, 3, 1, 2, 1, 3, 2, 3, 2, 3, 3, 2, 2, 2, 3, 2, 1, 2, 3, 1, 2, 3, 2, 3, 1, 2, 3, 2, 3],
    "equipment_age": [10, 12, 15, 8, 5, 3, 7, 9, 6, 11, 14, 6, 12, 8, 7, 5, 14, 16, 10, 9, 8, 12, 9, 11, 5, 8, 7, 9, 10, 8, 14, 16, 12, 6, 9, 7],
    "environment_temp": [40, 35, 50, 30, 60, 55, 45, 50, 35, 30, 60, 40, 50, 30, 55, 45, 50, 60, 40, 30, 55, 50, 45, 35, 55, 50, 60, 40, 30, 55, 45, 40, 30, 55, 50, 25]
}

# Data augmentation: synonym replacement
def synonym_replacement(text, n=1):
    words = text.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words if wordnet.synsets(word)]))
    
    if len(random_word_list) == 0:
        return text

    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = wordnet.synsets(random_word)
        synonym_words = list(set([syn.lemmas()[0].name() for syn in synonyms]))
        if len(synonym_words) > 0:
            synonym = random.choice(synonym_words)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break

    augmented_text = ' '.join(new_words)
    return augmented_text

# Preprocess text function
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r'\W', ' ',  text).lower()  # Replace non-word characters with space
    text = text.lower()  # Convert to lowercase
    text = contractions.fix(text)
    text = str(TextBlob(text).correct())
    tokens = [word for word in text.split() if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Create DataFrame and calculate RPN
df = pd.DataFrame(new_data)
df['RPN'] = df['severity'] * df['occurrence'] * df['detection'] / 1000  # Normalized RPN for scale

# Preprocess text data
df['description'] = df['description'].apply(preprocess_text)

# Data Augmentation
df_augmented = df.copy()
df_augmented['description'] = df_augmented['description'].apply(lambda x: synonym_replacement(x, n=1))
df = pd.concat([df, df_augmented], ignore_index=True)

# TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=20)
X_text = tfidf.fit_transform(df['description'].values).toarray()

# Combine numeric features for modeling
numeric_features = df[['severity', 'occurrence', 'detection', 'total_downtime', 'timeframe_to_fix', 'equipment_age', 'environment_temp']].values
scaler = StandardScaler()
numeric_features_scaled = scaler.fit_transform(numeric_features)

# Define the embedding size and max vocabulary size
EMBEDDING_DIM =100
MAX_VOCAB_SIZE = 10000
MAX_SEQ_LEN = 50

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(df['description'])
X_text_seq = tokenizer.texts_to_sequences(df['description'])
X_text_seq = pad_sequences(X_text_seq, maxlen=MAX_SEQ_LEN)

# Load pre-trained GloVe embeddings
def load_glove_embeddings(file_path, vocab_size, embedding_dim):
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

embeddings_index = load_glove_embeddings('glove.6B.100d.txt', MAX_VOCAB_SIZE, EMBEDDING_DIM)
word_index = tokenizer.word_index
num_words = min(MAX_VOCAB_SIZE, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i < MAX_VOCAB_SIZE:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            
# Define target (RPN)
rpn = df['RPN'].values

# Train-Test Split
X_train_text, X_test_text, X_train_num, X_test_num, y_train, y_test = train_test_split(
    X_text_seq, numeric_features_scaled, df['RPN'].values, test_size=0.3, random_state=42
)

# Define Model Architecture
text_input = Input(shape=(MAX_SEQ_LEN,), name='text_input')
embedding_layer = Embedding(num_words, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQ_LEN, trainable=False)(text_input)

gru_layer = Bidirectional(GRU(128, return_sequences=False, kernel_regularizer=l2(0.001)))(embedding_layer)

numeric_input = Input(shape=(7,), name='numeric_input')
concat_layer = concatenate([gru_layer, numeric_input])

# Adding Dense Layers with Batch Normalization and Dropout
dense_1 = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(concat_layer)
batch_norm_1 = BatchNormalization()(dense_1)
dropout_1 = Dropout(0.4)(batch_norm_1)

dense_2 = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(dropout_1)
batch_norm_2 = BatchNormalization()(dense_2)
dropout_2 = Dropout(0.4)(batch_norm_2)

dense_3 = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(dropout_2)
output_layer = Dense(1)(dense_3)

# Compile the Model
model = Model(inputs=[text_input, numeric_input], outputs=output_layer)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Learning Rate Scheduler
def step_decay(epoch):
    initial_lr = 0.001
    drop = 0.5
    epochs_drop = 10
    lr = initial_lr * (drop ** np.floor((1 + epoch) / epochs_drop))
    return lr

lr_scheduler = LearningRateScheduler(step_decay)

callbacks = [
    ModelCheckpoint('issue_predictor_model.keras', save_best_only=True, monitor='val_loss'),
    ReduceLROnPlateau(factor=0.2, patience=5, min_lr=1e-6),
    lr_scheduler
]

# Train the Model
history = model.fit(
    [X_train_text, X_train_num],
    y_train,
    validation_data=([X_test_text, X_test_num], y_test),
    epochs=100,
    batch_size=16,
    callbacks=callbacks,
    shuffle=True,
    verbose=1
)

# Plot Loss
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

# Model Evaluation
train_preds = model.predict([X_train_text, X_train_num])
test_preds = model.predict([X_test_text, X_test_num])

# Metrics Calculation
train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
train_mae = mean_absolute_error(y_train, train_preds)
test_mae = mean_absolute_error(y_test, test_preds)

logger.info(f"Train RMSE: {train_rmse:.4f}, Train MAE: {train_mae:.4f}")
logger.info(f"Test RMSE: {test_rmse:.4f}, Test MAE: {test_mae:.4f}")

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
numeric_feature_names = ['severity', 'occurrence', 'detection', 'total_downtime', 'timeframe_to_fix', 'equipment_age', 'environment_temp']
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
model.save('issue_predictor_model.keras')
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

logger.info("Model and scalers saved successfully.")