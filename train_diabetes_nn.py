# -*- coding: utf-8 -*-
"""
Created on Sun May 25 00:20:11 2025

@author: ASUS
"""

# -*- coding: utf-8 -*-

# -------------------------------------------------------------
# This file trains your Advanced Diabetes Prediction Model (Neural Network)
# and saves it for your Streamlit app.
# -------------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf # For the Neural Network
from tensorflow.keras.models import Sequential # To build the layers of the NN
from tensorflow.keras.layers import Dense # For the neural network layers
import pickle # To save the scaling rules

print("--- Starting to build your Advanced Diabetes Brain ---")

# --- Step 1: Load Your Diabetes Data ---
# Make sure 'diabetes.csv' is in the SAME FOLDER as this Python file!
try:
    data = pd.read_csv('diabetes.csv')
    print("Diabetes data loaded successfully from 'diabetes.csv'.")
    print(f"First 5 rows of your data:\n{data.head()}")
    print(f"Columns in your data: {data.columns.tolist()}")
except FileNotFoundError:
    print("ERROR: 'diabetes.csv' not found! Please ensure it's in the same folder as this script.")
    print("Cannot proceed without data. Exiting.")
    exit()
except Exception as e:
    print(f"ERROR loading diabetes.csv: {e}. Please check your CSV file.")
    exit()


# --- Step 2: Prepare Your Data for the Neural Network ---
# X will be all the input features (like Glucose, BMI, Age)
# y will be the 'Outcome' (0 for no diabetes, 1 for diabetes)
# This assumes 'Outcome' is the last column or the column containing 0s and 1s for diabetes.
# If your outcome column has a different name, change 'Outcome' below.
try:
    X = data.drop('Outcome', axis=1) # All columns EXCEPT 'Outcome'
    y = data['Outcome']              # Only the 'Outcome' column
    print(f"\nFeatures (X) selected: {X.columns.tolist()}")
    print(f"Target (y) selected: Outcome")
except KeyError:
    print("ERROR: 'Outcome' column not found in your 'diabetes.csv'!")
    print("Please check your CSV file's column names and update the code if needed.")
    exit()
except Exception as e:
    print(f"ERROR preparing data: {e}. Check your CSV structure.")
    exit()


# Split data into training and testing sets (80% for training, 20% for testing)
# Training data is used to teach the model. Testing data is used to see how well it learned on new, unseen data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data split: {len(X_train)} samples for training, {len(X_test)} for testing.")

# Scale your data (SUPER important for Neural Networks!)
# This makes all your numbers similar in range, so the model learns better and faster.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # Learn how to scale on training data
X_test_scaled = scaler.transform(X_test)       # Apply the same scaling rules to test data
print("Data scaled successfully.")


# --- Step 3: Build the Neural Network Model (Your Advanced Brain!) ---
# This is like stacking LEGO bricks!
model = Sequential([
    # First layer: Takes your input features (number of columns in X),
    # processes them into 12 "thoughts" (neurons), uses 'relu' activation for non-linearity.
    Dense(12, activation='relu', input_dim=X_train_scaled.shape[1]),
    # Second (hidden) layer: Takes those 12 thoughts, processes them into 8 new "thoughts".
    Dense(8, activation='relu'),
    # Output layer: Takes those 8 thoughts, and gives ONE final answer
    # (a probability between 0 and 1, like 0.1 for non-diabetic, 0.9 for diabetic).
    # 'sigmoid' activation is perfect for 0 or 1 prediction problems.
    Dense(1, activation='sigmoid')
])
print("Neural Network model built with 3 layers.")

# --- Step 4: Teach the Model How to Learn ---
# 'adam' is a very smart way for the model to optimize itself (find the best answers).
# 'binary_crossentropy' is how it measures its mistakes for problems where the answer is 0 or 1.
# 'accuracy' is what we want to see (how many correct predictions it makes).
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("Model compiled and ready for training.")


# --- Step 5: Train the Model (The Learning Part!) ---
# This might take a moment.
# 'epochs' means how many times the model looks at ALL the training data. More epochs = more learning.
# 'batch_size' means how many pieces of data it looks at before making a small adjustment.
print(f"\nTraining the model for 150 epochs (this might take a moment)...")
model.fit(X_train_scaled, y_train, epochs=150, batch_size=10, verbose=0) # verbose=0 means no detailed output during training
print("Model training finished!")


# --- Step 6: Test the Model (How well did it learn?) ---
# We use the separate 'test' data that the model has never seen before.
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Model tested! Its Accuracy on new data is: {accuracy*100:.2f}%")


# --- Step 7: Save Your New Advanced Brain and its Scaling Rules ---
# We save the trained Neural Network model itself in a special '.h5' format.
model_filename = 'diabetes_neural_network_model.h5'
model.save(model_filename) # Saves the entire model

# We also need to save the 'scaler' because when you predict new data in your app,
# you MUST scale it the EXACT SAME WAY you scaled the training data!
scaler_filename = 'diabetes_scaler.pkl'
with open(scaler_filename, 'wb') as f:
    pickle.dump(scaler, f)

print(f"\nNeural Network model saved as: {model_filename}")
print(f"Scaler (scaling rules) saved as: {scaler_filename}")

print("\n----------------------------------------------------------------------")
print("FINISHED MAKING YOUR ADVANCED DIABETES BRAIN!")
print("You should now see two new files in your project folder:")
print(f"- {model_filename}")
print(f"- {scaler_filename}")
print("These are your new 'Advanced Brain' and its 'Rulebook' for the app.")
print("----------------------------------------------------------------------")