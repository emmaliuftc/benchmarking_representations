import numpy as np
import os
import pandas as pd
import glob
import random
import warnings
from skimage import io, measure, morphology

# 1. Define where your data is and where the manifest should go
data_dir = "/scratch/mola/eml057/cytodl_workspace/data/fresh_data_june/"

# 2. Find all your point cloud files (change .csv to .ply or .txt if needed)
files = glob.glob(os.path.join(data_dir, "*.tif"))
files.sort()

# 3. Create the data dictionary
data = []

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


def analyze_3d_segmentation(segmentation_array):
    """
    Analyzes a 3D segmentation array for border touches, overall dimensions, 
    and the dimensions of the smallest bounding box enclosing the foreground.
    """
    # Convert to a binary mask to treat all foreground (>0) as a single True region.
    mask = segmentation_array > 0
    
    # --- 1. Check if any border voxel is True ---
    touches_z = np.any(mask[0, :, :]) or np.any(mask[-1, :, :])
    touches_y = np.any(mask[:, 0, :]) or np.any(mask[:, -1, :])
    touches_x = np.any(mask[:, :, 0]) or np.any(mask[:, :, -1])
    
    border_has_true = touches_z or touches_y or touches_x

    # --- 2. Calculate the dimensions of the array ---
    # Using .shape returns a tuple of the array's dimensions: (Z, Y, X)
    array_dimensions = mask.shape

    # --- 3. Calculate the dimensions of the smallest bounding box ---
    props = measure.regionprops(mask.astype(int))
    
    if props:
        # props[0] gets our single foreground region.
        # In 3D, bbox format is: (min_z, min_y, min_x, max_z, max_y, max_x)
        bbox = props[0].bbox
        
        # Calculate the length of the bounding box along each axis
        z_len = bbox[3] - bbox[0]
        y_len = bbox[4] - bbox[1]
        x_len = bbox[5] - bbox[2]
        
        # Store as a tuple of dimensions instead of multiplying them together
        bbox_dimensions = (z_len, y_len, x_len)
    else:
        # Fallback if the array is entirely composed of False/0
        bbox_dimensions = (0, 0, 0) 

    return {
        "border_has_true": border_has_true,
        "array_dimensions": array_dimensions,
        "bounding_box_dimensions": bbox_dimensions,
        "bbox": bbox
    }



for f in files:
    # Extract a clean ID from the filename
    
    file_id = os.path.basename(f).split('.')[0]

    image = io.imread(f)

    mask = fill(image)

    results = analyze_3d_segmentation(mask)

    print(f"Analysis Results for {file_id}:")
    print(f"1. Border voxel is True: {results['border_has_true']}")
    print(f"2. Total size of array:  {results['array_dimensions']} voxels")
    print(f"3. Bounding box size:    {results['bounding_box_dimensions']} voxels")
    print(f"4. Bounding box:         {results['bbox']}")