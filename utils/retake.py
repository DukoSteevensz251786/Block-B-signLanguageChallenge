import numpy as np
import pandas as pd

retake = 'B'
df = pd.read_csv(r'data/processed/hand_signs.csv')
df = df[df['label'] != retake]
df.to_csv(r'data/processed/hand_signs.csv', index=False)
print("Dropped rows with label ", retake)
