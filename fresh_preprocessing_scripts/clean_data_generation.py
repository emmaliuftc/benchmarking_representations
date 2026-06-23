import os
import glob
import json
import random
import numpy as np
import shutil

# Define paths
input_dir = "/scratch/mola/eml057/cytodl_workspace/data/point_clouds/"
output_dir = "/scratch/mola/eml057/cytodl_workspace/data/shapenet_format/"
dummy_class_id = "00000000"

def convert_to_shapenet():
    # Create the base output directory and class directory
    class_dir = os.path.join(output_dir, dummy_class_id)
    os.makedirs(class_dir, exist_ok=True)
    
    # Find all surface files
    surface_files = glob.glob(os.path.join(input_dir, "*_surface_8192.npy"))
    
    valid_ids = []
    
    for surface_path in surface_files:
        file_id = os.path.basename(surface_path).split('_')[0]
        sdf_path = os.path.join(input_dir, f"{file_id}_sdf_20000.npy")
        
        if not os.path.exists(sdf_path):
            print(f"Skipping {file_id}: SDF file not found.")
            continue
            
        # Create instance directory
        instance_dir = os.path.join(class_dir, file_id)
        os.makedirs(instance_dir, exist_ok=True)
        
        # Load the .npy arrays
        surface_arr = np.load(surface_path)
        sdf_arr = np.load(sdf_path)
        
        # Save as .npz (ShapeNet dataloaders usually look for the 'points' key inside the .npz)
        np.savez(os.path.join(instance_dir, "pointcloud.npz"), points=surface_arr)
        
        # For SDF, it's safer to separate the distances from the coordinates if it's the standard dataloader, 
        # or just save the 4D array if that's what cyto-dl expects. 
        # Here we save both the raw points and the split version just in case.
        np.savez(
            os.path.join(instance_dir, "points.npz"), 
            points=sdf_arr[:, 1:4],  # XYZ coords
            sdf=sdf_arr[:, 0],       # SDF values
            raw_4d=sdf_arr           # The combined 4D array
        )
        
        valid_ids.append(file_id)

    print(f"Successfully converted {len(valid_ids)} instances.")

    # Create Train (80%) / Val (10%) / Test (10%) Splits
    random.seed(42)
    random.shuffle(valid_ids)
    
    train_split = int(len(valid_ids) * 0.8)
    val_split = int(len(valid_ids) * 0.9)
    
    splits = {
        "train": valid_ids[:train_split],
        "val": valid_ids[train_split:val_split],
        "test": valid_ids[val_split:]
    }
    
    # Save split JSONs in the format ShapeNetDataModule expects
    for split_name, instances in splits.items():
        # Format: {"class_id": ["instance_1", "instance_2", ...]}
        split_dict = {dummy_class_id: instances}
        
        with open(os.path.join(output_dir, f"{split_name}.json"), 'w') as f:
            json.dump(split_dict, f, indent=4)
            
    print("Generated train.json, val.json, and test.json")

if __name__ == "__main__":
    convert_to_shapenet()