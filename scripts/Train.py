import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import pickle
import os

# Path to your collected data
CSV_PATH = 'data/processed/hand_signs.csv'

# Load the data
df = pd.read_csv(CSV_PATH)

# Features = all columns except the first (label)
X = df.iloc[:, 1:].values          # shape: (n_samples, 63)
y = df['label'].values             # shape: (n_samples,)

print(f"Loaded {len(df)} samples")
print("Classes found:", sorted(np.unique(y)))

# Optional: filter to only keep classes with enough samples
min_samples_per_class = 20
class_counts = df['label'].value_counts()
good_classes = class_counts[class_counts >= min_samples_per_class].index
df = df[df['label'].isin(good_classes)]
print(f"After filtering: {len(df)} samples")



X = df.iloc[:, 1:].values
y = df['label'].values

# Split into train / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")

# Train a Random Forest
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.3f} ({acc*100:.1f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
MODEL_PATH = os.path.join('models', 'sign_model_rf.pkl')
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)
print(f"Model saved to: {MODEL_PATH}")