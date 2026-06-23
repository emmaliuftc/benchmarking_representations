import numpy as np
import os
import pandas as pd
import glob
import random
import warnings
from skimage import io, measure, morphology
from stl import mesh

data_dir = "/scratch/mola/eml057/cytodl_workspace/data/fresh_data_june/"

files = glob.glob(os.path.join(data_dir, "*.tif"))
files.sort()


def fill(x):
    newfile = np.zeros_like(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        for i in range(len(x)):
            roi = x[i, :, :]
            new = morphology.remove_small_holes(roi, area_threshold=1000)
            newfile[i, :, :] = new
        newfile = morphology.remove_small_holes(newfile, area_threshold=100000)
        newfile = morphology.remove_small_objects(newfile, min_size=5000)
    return newfile

for f in files:
    # Extract a clean ID from the filename
    file_id = os.path.basename(f).split('.')[0]

    image = io.imread(f)

    mask = fill(image)
    print(mask.shape)
    print("Padding to center in cube...")
    
    # 1. Find the largest dimension to determine the cube size
    max_dim = max(mask.shape)
    
    # 2. Build the padding configuration for each axis
    pad_width = []
    for current_dim in mask.shape:
        # Calculate total padding needed for this axis
        pad_total = max_dim - current_dim
        
        # Split the padding evenly to center the array
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        
        pad_width.append((pad_before, pad_after))
        
    # 3. Pad the array with 0s
    padded_mask = np.pad(
        mask, 
        pad_width=pad_width, 
        mode='constant', 
        constant_values=0
    )
    print(padded_mask.shape)

    # Do extra bounds in case the largest dimension has something touching the side
    padded_mask = np.pad(padded_mask, pad_width=1, mode="constant",constant_values=0)
    print(padded_mask.shape)

    print("------")





    if np.min(padded_mask) == np.max(padded_mask):
        print("skip marching cubes because no surface")
        continue
    else:
        verts, faces, normals, values = measure.marching_cubes(padded_mask, step_size=1)

    # Create the mesh object for numpy-stl
    # Each row in the vectors array represents a triangular face with 3 vertices
    stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            stl_mesh.vectors[i][j] = verts[face[j], :]

    # 4. Save the mesh to an STL file
    stl_mesh.save(f"/scratch/mola/eml057/cytodl_workspace/data/stl_fresh/{file_id}.stl")
    print(f"Successfully saved to .../data/stl_fresh/{file_id}.stl")