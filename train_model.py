import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def train_lstm_model():
    print("Loading dataset...")
    try:
        df = pd.read_csv('dataset.csv')
    except FileNotFoundError:
        print("dataset.csv not found!")
        return

    # Filter for valid diseases (Goal: Predict Disease based on Article Content)
    print("Filtering data...")
    # Remove rows where Disease is Unknown or missing
    df_clean = df[df['Maladie'].notna()]
    df_clean = df_clean[df_clean['Maladie'] != 'Unknown']
    
    # Check if we have enough data
    if len(df_clean) < 10:
        print(f"Warning: Only {len(df_clean)} samples found with known diseases.")
        print("Switching target to 'Source' classification as a fallback.")
        df_clean = df # Use full dataset
        target = 'Source'
    else:
        target = 'Maladie'
        print(f"Targeting '{target}' with {len(df_clean)} samples.")

    X_text = df_clean['Contenu'].astype(str).tolist()
    y_labels = df_clean[target].tolist()

    # Preprocessing Parameters
    vocab_size = 2000
    embedding_dim = 64
    max_length = 200
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"

    # Tokenization
    print("Tokenizing text...")
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(X_text)
    sequences = tokenizer.texts_to_sequences(X_text)
    padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    # Label Encoding
    print(f"Encoding labels for '{target}'...")
    label_encoder = LabelEncoder()
    training_labels = label_encoder.fit_transform(y_labels)
    num_classes = len(np.unique(training_labels))
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {label_encoder.classes_}")

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(padded, training_labels, test_size=0.2, random_state=42)

    # Model Definition: LSTM
    # LSTM is chosen because it handles sequential text data better than simple Dense nets.
    print("Building LSTM Model...")
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim), # input_length removed
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # Model Training
    print("Starting training...")
    history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), verbose=1)

    print("\nTraining Complete.")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Final Test Accuracy: {accuracy*100:.2f}%")
    
    # Save the model
    model.save('disease_classifier_lstm.keras')
    print("Model saved to 'disease_classifier_lstm.keras'")
    
    # --- DEMONSTRATION: How to see the output ---
    print("\n--- DEMONSTRATION: PREDICTING NEW TEXT ---")
    sample_texts = [
        "Un cas de vache folle a été détecté dans une ferme.", # Should predict Vache Folle (or similar)
        "Des oiseaux morts de la grippe aviaire ont été trouvés."
    ]
    
    # Preprocess
    seqs = tokenizer.texts_to_sequences(sample_texts)
    padded_seqs = pad_sequences(seqs, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    
    # Predict
    predictions = model.predict(padded_seqs)
    predicted_indices = np.argmax(predictions, axis=1)
    predicted_labels = label_encoder.inverse_transform(predicted_indices)
    
    for i, text in enumerate(sample_texts):
        print(f"\nText: {text}")
        print(f"Predicted Disease: {predicted_labels[i]}")
        print(f"Confidence: {np.max(predictions[i])*100:.2f}%")

if __name__ == "__main__":
    try:
        train_lstm_model()
    except Exception as e:
        print(f"An error occurred: {e}")
