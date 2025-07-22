import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import joblib

# Save the fitted scaler


# Step 1: Load and shuffle data
df = pd.read_csv("X_all_gestures_labeled.csv")
df = shuffle(df, random_state=42)

# Step 2: Separate features and labels
X = df.drop(columns=["label"]).astype("float32")
y = df["label"]

# Step 3: Normalize features (0 to 1)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
joblib.dump(scaler, "models/scaler.pkl")
print("Scaler saved to models/scaler.pkl")
# Step 4: Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(np.unique(y_encoded))
y_categorical = to_categorical(y_encoded, num_classes=num_classes)

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42
)

# Step 6: Define the improved model
model = Sequential([
    Dense(256, activation='relu', input_shape=(X.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

# Step 7: Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 8: Train with early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_split=0.1, callbacks=[early_stop])

# Step 9: Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"\n Test Accuracy: {acc * 100:.2f}%")

# Step 10: Save the model
os.makedirs("models", exist_ok=True)
model.save("models/gesture_classifier.h5")  # modern format
print(" Model saved to models/gesture_classifier.h5")
