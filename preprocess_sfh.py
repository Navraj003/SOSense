import pandas as pd

# Load full dataset
df = pd.read_csv("datasets/hand_landmarks.csv")

# Show unique gesture labels
print("Unique gesture labels:", df["sign"].unique())

# Drop unwanted metadata columns (if any)
X = df.drop(columns=["sign", "sex", "user", "hand"], errors="ignore")
y = df["sign"]  # Labels

# Save features and labels for training
X.to_csv("X_all_gestures.csv", index=False)
y.to_csv("y_all_gestures.csv", index=False)

print(f"Saved features and labels for all 7 gestures.")
