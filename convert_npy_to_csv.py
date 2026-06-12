import numpy as np
import pandas as pd
import glob

# Find all point cloud .npy files in your processed_data folder
npy_files = glob.glob("./processed_data/*_ptcloud.npy")

print(f"Found {len(npy_files)} files to convert...")

for file in npy_files:
    # Load the numpy array (assuming it's an N x 3 array of coordinates)
    xyz = np.load(file)

    # Convert to a pandas DataFrame with strict 3D headers that PyntCloud expects
    df = pd.DataFrame(xyz, columns=['x', 'y', 'z'])

    # Save as CSV
    csv_path = file.replace('.npy', '.csv')
    df.to_csv(csv_path, index=False)

print("Conversion complete!")
