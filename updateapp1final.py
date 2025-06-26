import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dropout, Dense, BatchNormalization
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.regularizers import l2
import tkinter as tk
from tkinter import simpledialog, messagebox

# Load dataset
df = pd.read_csv("movies3.csv")

# Handling NaNs and infinite values
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Define numerical and categorical columns
numerical_cols = ['Year', 'Rating', 'Votes']
categorical_cols = ['Genre', 'Language', 'Movie Name']

# Fill missing values in numerical columns with mean
df[numerical_cols] = df[numerical_cols].apply(pd.to_numeric, errors='coerce')
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

# Fill missing values in categorical columns
df['Genre'] = df['Genre'].fillna('Unknown')
df['Language'] = df['Language'].fillna('Unknown')
df['Movie Name'] = df['Movie Name'].fillna('Unknown')

# Fix 'Timing' column - Convert "134 min" -> 134, and handle non-numeric values
df['Timing'] = df['Timing'].astype(str).str.replace(',', '')  # Remove commas
df['Timing'] = df['Timing'].str.extract('(\d+)')  # Extract numeric part
df['Timing'] = df['Timing'].astype(float).fillna(0).astype(int)

# Encode categorical features
genre_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
genre_encoded = genre_encoder.fit_transform(df[['Genre']])

language_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
language_encoded = language_encoder.fit_transform(df[['Language']])

# Combine numerical and encoded features
X_numerical = df[numerical_cols].values
X_combined = np.concatenate([X_numerical, genre_encoded, language_encoded], axis=1)

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_combined)

# Simulate interaction data
interactions = pd.DataFrame({
    'user_id': np.random.randint(1, 100, 1000),
    'movie_id': np.random.choice(df.index, 1000),  # Fixed KeyError issue
    'timestamp': pd.date_range(start='2020-01-01', periods=1000, freq='h')
})

# Merge interaction data with movie data
interactions = interactions.merge(df, left_on='movie_id', right_index=True)
interactions = interactions.merge(pd.DataFrame(X_scaled, columns=[f'feature_{i}' for i in range(X_scaled.shape[1])]), left_index=True, right_index=True)
interactions = interactions.sort_values(by=['user_id', 'timestamp'])

# Function to create sequences
def create_sequences(interactions, max_sequence_length, feature_dim):
    sequences = []
    for user_id, group in interactions.groupby('user_id'):
        user_sequences = group.iloc[:, -feature_dim:].values  # Ensure only feature columns are selected
        for i in range(1, len(user_sequences)):
            seq = user_sequences[max(0, i - max_sequence_length):i]
            
            # Fix shape inconsistency
            if seq.shape[1] != feature_dim:
                continue  # Skip sequences with mismatched dimensions

            sequences.append(seq)
    return sequences

max_sequence_length = 20
feature_dim = X_scaled.shape[1]
sequences = create_sequences(interactions, max_sequence_length, feature_dim)

# Prepare data for training
X = np.zeros((len(sequences), max_sequence_length, feature_dim), dtype=np.float32)
y = np.zeros((len(sequences), max_sequence_length), dtype=np.int32)

for i, seq in enumerate(sequences):
    seq_length = min(len(seq), max_sequence_length)
    X[i, :seq_length, :] = seq[:seq_length]  # Assign only valid values

    # Optional: Pad remaining positions with the last known value
    if seq_length < max_sequence_length:
        X[i, seq_length:, :] = seq[-1]  # Repeat last known sequence value

# Convert y to one-hot encoding
y_train_onehot = to_categorical(y, num_classes=len(df) + 1).astype(np.float32)  # ✅ FIXED OUTPUT SHAPE

# Define and compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)

model = Sequential([
    GRU(units=128, input_shape=(max_sequence_length, X_scaled.shape[1]), return_sequences=True, kernel_initializer='he_normal', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.3),
    GRU(units=128, return_sequences=True, kernel_initializer='he_normal', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(units=len(df) + 1, activation='softmax')  # ✅ FIXED OUTPUT SHAPE
])

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y_train_onehot, test_size=0.2, random_state=42)

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=64,  # Reduced batch size to prevent memory issues
    validation_data=(X_test, y_test),
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)]
)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Movie recommendation function (with random user_id)
def recommend_movies(movie_name, top_n=10):
    movie_row = df[df['Movie Name'].str.strip().str.lower() == movie_name.strip().lower()]
    if movie_row.empty:
        print(f"Movie '{movie_name}' not found.")
        return []

    # Encode categorical features for the movie
    genre_encoded = genre_encoder.transform(movie_row[['Genre']])
    language_encoded = language_encoder.transform(movie_row[['Language']])
    
    # Combine numerical and encoded categorical features
    movie_features = np.concatenate([movie_row[numerical_cols].values, genre_encoded, language_encoded], axis=1)
    
    # Scale the features using the same scaler
    movie_features_scaled = scaler.transform(movie_features)
    
    # Pick an existing user instead of a random one
    existing_users = interactions['user_id'].unique()
    user_id = random.choice(existing_users)

    # Get past movies the user has watched
    past_movies = interactions[interactions['user_id'] == user_id]['movie_id'].values[-max_sequence_length:]
    
    # Create input sequence
    user_sequence = np.zeros((1, max_sequence_length, feature_dim))
    for i, past_movie_id in enumerate(past_movies):
        user_sequence[0, i, :] = X_scaled[past_movie_id]

    # Predict movie preferences
    preferences = model.predict(user_sequence)[0][-1]

    # ✅ Ensure probabilities match the movie dataset size
    if len(preferences) > len(df):
        preferences = preferences[:len(df)]  # Trim excess values
    elif len(preferences) < len(df):
        padding = np.zeros(len(df) - len(preferences))
        preferences = np.concatenate([preferences, padding])  # Pad missing values

    # Apply softmax sampling
    probabilities = np.exp(preferences) / np.sum(np.exp(preferences))

    # ✅ Ensure probabilities sum to 1
    probabilities /= probabilities.sum()

    # Select top recommended movies
    top_indices = np.random.choice(len(df), size=min(top_n, len(df)), p=probabilities, replace=False)

    return df.iloc[top_indices]['Movie Name'].values


# GUI for user input
def get_movie_name():
    while True:
        movie_name = simpledialog.askstring("Movie Recommendation", "Enter the movie name (or type 'exit' to quit):")
        if not movie_name or movie_name.lower() == 'exit':
            break
        
        recommended_movies = recommend_movies(movie_name, top_n=10)
        if recommended_movies.any():
            messagebox.showinfo("Recommended Movies", "\n".join(recommended_movies))
        else:
            messagebox.showerror("Error", f"No recommendations found for '{movie_name}'")

# Run GUI
root = tk.Tk()
root.withdraw()
get_movie_name()
root.destroy()