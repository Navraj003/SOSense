import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get file paths from environment variables
FEATURES_PATH = os.getenv('FEATURES_PATH', 'X_all_gestures_labeled.csv')

df = pd.read_csv(FEATURES_PATH)

if 'label' in df.columns:
    print("\nLabel distribution:")
    print(df['label'].value_counts())
else:
    print("❌ 'label' column not found.")
    print("Columns available:", df.columns.tolist())
