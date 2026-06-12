import pandas as pd

# 1. Load your dataset manifest
file_name = "neutrophil_sdf_manifest.csv"  # Update this to your actual filename
df = pd.read_csv(file_name)

# 2. Strip any invisible spaces or weird characters from the headers
df.columns = df.columns.str.strip()

# 3. Force the sdf column to be named exactly 'sdf'
if 'sdf_path' in df.columns:
    df = df.rename(columns={'sdf_path': 'sdf'})

# 4. Save it safely
df.to_csv(file_name, index=False)
print("CSV is completely sanitized. Active headers:", df.columns.tolist())
