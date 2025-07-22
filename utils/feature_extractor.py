# utils/feature_extractor.py

import numpy as np

def extract_features(landmarks):
    """
    Given MediaPipe landmarks (21x3), return 64-dim feature vector.
    """
    arr = np.array(landmarks)  # (21, 3)
    flat = arr.flatten().tolist()  # 63 values

    # Example: add distance between wrist (0) and index tip (8)
    distance = np.linalg.norm(arr[0] - arr[8])
    flat.append(distance)

    return np.array(flat, dtype=np.float32)
