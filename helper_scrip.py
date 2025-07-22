import pandas as pd

df = pd.read_csv("X_all_gestures_labeled.csv")

if 'label' in df.columns:
    print("\nLabel distribution:")
    print(df['label'].value_counts())
else:
    print("❌ 'label' column not found.")
    print("Columns available:", df.columns.tolist())
