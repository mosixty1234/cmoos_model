# Import necessary libraries for data manipulation, model building, and evaluation.
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
from sklearn.ensemble import GradientBoostingRegressor
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
import os
import gdown
from textblob import TextBlob

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

#Set up logging to monitor model training and evaluation processe
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#  Define a dictionary of common mechanical issues and their recommended solutions for better interpretability.
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

#  Create a new dataset of machine issues to improve the model's robustness by including diverse scenarios
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

# Create a DataFrame from the new data and calculate the Risk Priority Number (RPN) to quantify risks associated with equipment issues
df = pd.DataFrame(new_data)
df['RPN'] = df['severity'] * df['occurrence'] * df['detection'] / 1000  # Normalized RPN for scale

# Preprocess the description text to improve input quality for the model.
df['description'] = df['description'].apply(preprocess_text)

# Augment the dataset with new descriptions to ensure a richer training set for better model performance
df_augmented = df.copy()
df_augmented['description'] = df_augmented['description'].apply(lambda x: synonym_replacement(x, n=1))
df = pd.concat([df, df_augmented], ignore_index=True)

# TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=20)
X_text = tfidf.fit_transform(df['description'].values).toarray()

# Combine numeric features with text data for comprehensive input to the mode
numeric_features = df[['severity', 'occurrence', 'detection']].values
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

# Ensure GloVe file is downloaded
GLOVE_PATH = 'glove.6B.100d.txt' 

if not os.path.exists(GLOVE_PATH):
    file_id = '1GiBauOchTROVKnRMtfGAjtiTC55apMgg'  # Replace with your actual file ID if necessary
    gdown.download(f'https://drive.google.com/uc?id={file_id}', GLOVE_PATH, quiet=False)

# Load pre-trained GloVe embeddings to enhance the model's understanding of textual data.
def load_glove_embeddings(file_path, embedding_dim):
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index


#
print("Loading GloVe embeddings...")
embeddings_index = load_glove_embeddings(GLOVE_PATH, EMBEDDING_DIM)
print(f"Found {len(embeddings_index)} word vectors.")

# Set parameters for embedding and tokenization, crucial for handling the text input effectivel
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
texts = [
"Overheating in the motor due to prolonged usage.",
    "Corrosion detected in the water pump affecting performance.",
    "High vibrations in the conveyor system causing belt misalignment.",
    "Unexpected power outage leading to sudden equipment failure.",
    "Software malfunction causing incorrect output data.",
    "Oil leakage in the hydraulic system reducing pressure.",
    "Unusual noise from the gearbox indicating potential bearing failure.",
    "Electrical short circuit causing frequent tripping.",
    "Sensor calibration issues leading to inaccurate measurements.",
    "Cooling system failure causing excessive heat buildup.",
    "Pump cavitation due to low fluid levels.",
    "Valve malfunction preventing proper closure of the system.",
    "Inconsistent readings from the pressure sensor.",
    "Motor bearings overheating, causing loud grinding noises.",
    "Compressor failure resulting in reduced output pressure.",
    "Hydraulic fluid contamination causing irregular actuator movement.",
    "Severe wear on conveyor belt leading to material loss.",
    "Fan motor failure resulting in insufficient cooling.",
    "Overloaded electrical circuit causing frequent breaker trips.",
    "Air filter clogging reducing airflow efficiency.",
    "Inconsistent temperature control in the refrigeration unit.",
    "Frequent jamming in the automated sorting machine.",
    "Leaking valve in the cooling system, reducing overall performance.",
    "Rotor imbalance in the motor causing high levels of vibration.",
    "High humidity leading to condensation buildup in electrical components.",
    "Worn-out actuator leading to delayed response in control systems.",
    "Inadequate lubrication in bearings causing overheating.",
    "Cracked impeller in the pump reducing flow rate.",
    "Excessive pressure build-up in the boiler system.",
    "Frequent system reboots caused by software bugs.",
    "Actuator sticking, resulting in slow valve operation.",
    "Loose wiring connections causing intermittent power failures.",
    "Bearing wear causing misalignment in the drive shaft.",
    "Temperature sensor fault leading to incorrect readings.",
    "Compressor overheating due to insufficient ventilation.",
    "Leak in the hydraulic system reducing system pressure.",
    "Overloaded gear mechanism causing slow movement.",
    "Clogged exhaust fan reducing air circulation in the system.",
    "Inconsistent fluid flow in the hydraulic lines.",
    "Worn bearings in the motor leading to excessive play.",
    "Insulation failure in electrical components causing shorts.",
    "Blocked vents leading to overheating of machinery.",
    "Excessive wear on conveyor rollers affecting operation.",
    "Inadequate power supply causing frequent resets.",
    "Failed circuit breaker due to overloaded circuits.",
    "Insufficient maintenance leading to equipment degradation.",
    "Unplanned downtime due to sudden equipment failure.",
    "Failures in the automated control system affecting efficiency.",
    "Issues with software updates causing compatibility problems.",
    "Malfunctioning pressure relief valve leading to safety concerns.",
    "Damage to hydraulic hoses causing fluid leaks.",
    "Frequent errors in data acquisition systems.",
    "Misalignment of drive components causing noise.",
    "Power supply fluctuations affecting sensitive equipment.",
    "Broken seals in pneumatic systems leading to air loss.",
    "Routine wear and tear causing operational inefficiencies.",
    "Overheating of bearings due to lack of lubrication.",
    "Contaminated hydraulic fluid causing valve stickiness.",
    "Irregularities in voltage supply affecting motor performance.",
    "Wear on rubber seals causing leaks in fluid systems.",
    "Unexpected fluctuations in load affecting stability.",
    "Electrical grounding issues causing erratic behavior.",
    "Damaged circuit boards leading to malfunctioning equipment.",
    "Improperly calibrated gauges giving false readings.", 
    "Increased wear on the gears due to misalignment.",
    "Frequent clogging of filters affecting fluid flow.",
    "Unusual fluctuations in motor speed during operation.",
    "Overloading of the main circuit causing overheating.",
    "Improper shutdown procedures leading to system errors.",
    "Vibration sensors indicating potential mechanical failure.",
    "Pneumatic actuator not responding to control signals.",
    "Deterioration of electrical insulation in wiring."]  # Replace this with your actual text data
tokenizer.fit_on_texts(texts)

# Convert text data to sequences and pad them
sequences = tokenizer.texts_to_sequences(texts)
X_text_seq = pad_sequences(sequences, maxlen=MAX_SEQ_LEN)

#Load pre-trained GloVe embeddings to enhance the model's understanding of textual data
word_index = tokenizer.word_index
num_words = min(MAX_VOCAB_SIZE, len(word_index) + 1)

# Prepare an embedding matrix that links words to their GloVe vectors for improved training performance
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i < MAX_VOCAB_SIZE:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            
# Define the target variable (RPN) which the model will learn to predict based on the feature
rpn = df['RPN'].values

#Split the dataset into training and testing sets to validate the model's performance on unseen data.
X_train_text, X_test_text, X_train_num, X_test_num, y_train, y_test = train_test_split(
    X_text_seq, numeric_features_scaled, df['RPN'].values, test_size=0.5, random_state=42
)

# Define the architecture of the model, combining text input and numerical features through a series of layers
text_input = Input(shape=(MAX_SEQ_LEN,), name='text_input')
embedding_layer = Embedding(num_words, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQ_LEN, trainable=False)(text_input)

gru_layer = Bidirectional(GRU(128, return_sequences=False, kernel_regularizer=l2(0.001)))(embedding_layer)

numeric_input = Input(shape=(3,), name='numeric_input')
concat_layer = concatenate([gru_layer, numeric_input])

# Implement dropout and batch normalization to improve model generalization and prevent overfitting
dense_1 = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(concat_layer)
batch_norm_1 = BatchNormalization()(dense_1)
dropout_1 = Dropout(0.5)(batch_norm_1)

dense_2 = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(dropout_1)
batch_norm_2 = BatchNormalization()(dense_2)
dropout_2 = Dropout(0.5)(batch_norm_2)

dense_3 = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(dropout_2)
output_layer = Dense(1)(dense_3)

# CCompile the model using an appropriate optimizer and loss function, focusing on minimizing prediction errors.
model = Model(inputs=[text_input, numeric_input], outputs=output_layer)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])


#Implement a learning rate scheduler to adjust the learning rate during training, promoting effective convergence.
def step_decay(epoch):
    initial_lr = 0.001
    drop = 0.5
    epochs_drop = 10
    lr = initial_lr * (drop ** np.floor((1 + epoch) / epochs_drop))
    return lr

lr_scheduler = LearningRateScheduler(step_decay)

#Define callbacks to save the best model and reduce the learning rate based on validation performance
callbacks = [
    ModelCheckpoint('issue_predictor_model.keras', save_best_only=True, monitor='val_loss'),
    ReduceLROnPlateau(factor=0.2, patience=5, min_lr=1e-05),
    lr_scheduler
]

# Train the model with the training data while validating its performance on a separate set.
history = model.fit(
    [X_train_text, X_train_num],
    y_train,
    validation_data=([X_test_text, X_test_num], y_test),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    shuffle=True,
    verbose=1
)

# Plot and visualize the loss during training to assess the model's learning progress.
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

#Evaluate the model on the test set and calculate performance metrics (RMSE and MAE) to quantify its accuracy.
train_preds = model.predict([X_train_text, X_train_num])
test_preds = model.predict([X_test_text, X_test_num])

# Log the performance metrics for further analysis and to track improvements over time
train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
train_mae = mean_absolute_error(y_train, train_preds)
test_mae = mean_absolute_error(y_test, test_preds)

logger.info(f"Train RMSE: {train_rmse:.4f}, Train MAE: {train_mae:.4f}")
logger.info(f"Test RMSE: {test_rmse:.4f}, Test MAE: {test_mae:.4f}")

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.show()

# Save the trained model and necessary scalers for future use
model.save('issue_predictor_model.keras')
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
logger.info("Model and scalers saved successfully.")