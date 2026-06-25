import numpy as np

# Point to one of your updated SDF files
file_path = "/scratch/mola/eml057/cytodl_workspace/data/shapenet_format/00000000/001/df.npy"

# Load the array
sdf_values = np.load(file_path)

print(f"--- SDF Stats ---")
print(f"Shape: {sdf_values.shape}")
print(f"Absolute Min: {sdf_values.min()}")
print(f"Absolute Max: {sdf_values.max()}")

# Count how many points hit the exact clipping threshold
num_clipped_pos = np.sum(sdf_values >= 2.0)
num_clipped_neg = np.sum(sdf_values <= -2.0)
total_points = len(sdf_values)

print(f"\n--- Clipping Evidence ---")
print(f"Points artificially capped at +2.0: {num_clipped_pos} ({(num_clipped_pos/total_points)*100:.1f}%)")
print(f"Points artificially capped at -2.0: {num_clipped_neg} ({(num_clipped_neg/total_points)*100:.1f}%)")