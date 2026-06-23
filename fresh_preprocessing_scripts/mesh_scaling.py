import numpy as np
import os
import pandas as pd
import glob
import random
import warnings
from skimage import io, measure, morphology
from stl import mesh
import trimesh

data_dir = "/scratch/mola/eml057/cytodl_workspace/data/stl_fresh/"

files = glob.glob(os.path.join(data_dir, "*.stl"))
files.sort()

target_size = 1e9

meshes = []

max_dim = -1

filenames = []

for f in files:
    # Extract a clean ID from the filename
    file_id = os.path.basename(f).split('.')[0]

    mesh = trimesh.load(f)
    bounds = mesh.bounds
    zextent = bounds[1][0] - bounds[0][0]
    xextent = bounds[1][1] - bounds[0][1]
    yextent = bounds[1][2] - bounds[0][2]
    print(f"{file_id}: {bounds}, {zextent}, {xextent}, {yextent}")
    
    cubic_dim = max(xextent,yextent,zextent)
    target_size = min(target_size, cubic_dim)
    max_dim = max(max_dim, cubic_dim)

    meshes.append(mesh)
    filenames.append(file_id)



# 1. Find the maximum dimension across the entire dataset
    
D_max_global = max_dim

# 2. Calculate the single Global Scaling Factor
S_global = target_size / D_max_global

print(f"Smallest largest original dimension: {target_size}")
print(f"Largest original dimension: {D_max_global}")
print(f"Global Scaling Factor: {S_global:.4f}")

# 3. Apply the scale and center each mesh
scaled_meshes = []
for mesh in meshes:
    # Find the center of the current mesh
    center = np.mean(mesh.vertices, axis=0)
    
    # Shift to the origin (0, 0, 0)
    centered_vertices = mesh.vertices - center
    
    # Apply the global scaling factor
    scaled_vertices = centered_vertices * S_global
    
    # # (Optional) Shift the mesh to the center of a [0, 32] grid 
    # # If your model expects coordinates from 0 to 32 instead of -16 to 16
    # offset = target_size / 2.0
    # final_vertices = scaled_vertices + np.array([offset, offset, offset])
    
    # Update and save the mesh
    mesh.vertices = scaled_vertices
    scaled_meshes.append(mesh)

print("SCALED MESH DATA: \n\n\n\n\n\n\n\n\n\n\n\n\n")
for i, mesh in enumerate(scaled_meshes):

    bounds = mesh.bounds
    zextent = bounds[1][0] - bounds[0][0]
    xextent = bounds[1][1] - bounds[0][1]
    yextent = bounds[1][2] - bounds[0][2]

    print(f"{filenames[i]}: {bounds}, {zextent}, {xextent}, {yextent}")

    mesh.export(f"/scratch/mola/eml057/cytodl_workspace/data/stl_scaled/{filenames[i]}.stl")
    print(f"Successfully saved to .../data/stl_scaled/{filenames[i]}.stl")