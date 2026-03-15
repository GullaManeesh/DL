import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

# Load dataset
data = fetch_california_housing()

X = data.data
y = data.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, "scaler.pkl")

# Build MLP Model
model = Sequential()

model.add(Dense(64, activation='relu', input_shape=(8,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
model.summary()

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# Train model
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Save model
model.save("model.h5")

print("Model and scaler saved")

# GRAPH 1: Training vs Validation Loss

plt.figure(figsize=(8,5))

plt.plot(history.history['loss'], color='blue', linewidth=2, label='Training Loss')
plt.plot(history.history['val_loss'], color='orange', linewidth=2, label='Validation Loss')

plt.title('Training vs Validation Loss', fontsize=14)
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)

plt.show()


# GRAPH 2: Actual vs Predicted Prices

y_pred = model.predict(X_test)

plt.figure(figsize=(8,5))

plt.scatter(y_test, y_pred, color='purple', alpha=0.5)

# perfect prediction line
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red', linewidth=2)

plt.title("Actual vs Predicted House Prices", fontsize=14)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.grid(True)

plt.show()