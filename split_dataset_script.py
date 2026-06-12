import pandas as pd
import numpy as np

# Load your manifest
df = pd.read_csv("./neutrophil_sdf_manifest.csv")

# Randomly assign 80% to train, 20% to val
df['split'] = np.random.choice(['train', 'val'], size=len(df), p=[0.8, 0.2])

# Save it back (overwriting the file)
df.to_csv("./neutrophil_sdf_manifest.csv", index=False)
print(
    f"Split added! Train: {len(df[df['split']=='train'])}, Val: {len(df[df['split']=='val'])}")
