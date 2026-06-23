# import json
# import os

# base_dir = "/scratch/mola/eml057/cytodl_workspace/data/shapenet_format/"
# class_id = "00000000"
# class_dir = os.path.join(base_dir, class_id)

# for split in ['train', 'val', 'test']:
#     json_path = os.path.join(base_dir, f"{split}.json")
#     lst_path = os.path.join(class_dir, f"{split}.lst")
    
#     if os.path.exists(json_path):
#         with open(json_path, 'r') as f:
#             data = json.load(f)
            
#         # Extract the list of instance IDs for this class
#         instances = data.get(class_id, [])
        
#         # Write them line-by-line to the .lst file
#         with open(lst_path, 'w') as f:
#             for inst in instances:
#                 f.write(f"{inst}\n")
                
#         print(f"Created {lst_path} with {len(instances)} instances.")




""" import os
import shutil
import glob

input_dir = "/scratch/mola/eml057/cytodl_workspace/data/point_clouds/"
base_dir = "/scratch/mola/eml057/cytodl_workspace/data/shapenet_format/00000000/"

# Find all original SDF .npy files
sdf_files = glob.glob(os.path.join(input_dir, "*_sdf_20000.npy"))

for sdf_path in sdf_files:
    # Extract the ID (e.g., "041" from "041_sdf_20000.npy")
    file_id = os.path.basename(sdf_path).split('_')[0]
    
    # Define the target instance directory
    instance_dir = os.path.join(base_dir, file_id)
    
    # If the folder exists from our previous script, copy the file in
    if os.path.exists(instance_dir):
        target_path = os.path.join(instance_dir, "df.npy")
        shutil.copy(sdf_path, target_path)
        
print("Successfully copied all SDF arrays as df.npy!")
 """


# import os
# import numpy as np

# base_dir = "/scratch/mola/eml057/cytodl_workspace/data/shapenet_format/00000000/"

# # Iterate through every instance folder
# for inst in os.listdir(base_dir):
#     inst_dir = os.path.join(base_dir, inst)
#     if not os.path.isdir(inst_dir): 
#         continue

#     # 1. Fix the .npz files (surface points and SDF coordinates)
#     for file_name in ["pointcloud.npz", "points.npz"]:
#         path = os.path.join(inst_dir, file_name)
#         if os.path.exists(path):
#             with np.load(path) as data:
#                 # Cast every array in the archive to float32
#                 new_data = {k: v.astype(np.float32) for k, v in data.items()}
#             # Overwrite the file
#             np.savez(path, **new_data)

#     # 2. Fix the standalone df.npy files (SDF distances)
#     df_path = os.path.join(inst_dir, "df.npy")
#     if os.path.exists(df_path):
#         df_arr = np.load(df_path)
#         np.save(df_path, df_arr.astype(np.float32))

# print("Successfully converted all point clouds to 32-bit floats!")





""" import os
import numpy as np

base_dir = "/scratch/mola/eml057/cytodl_workspace/data/shapenet_format/00000000/"

for inst in os.listdir(base_dir):
    inst_dir = os.path.join(base_dir, inst)
    if not os.path.isdir(inst_dir): 
        continue
        
    df_path = os.path.join(inst_dir, "df.npy")
    if os.path.exists(df_path):
        # Load the 4D array (20000, 4)
        df_arr = np.load(df_path)
        
        # If it's still 4D, slice it to keep ONLY the first column (the SDF values)
        if len(df_arr.shape) > 1 and df_arr.shape[1] == 4:
            # Slice column 0 and ensure it's a 32-bit float
            sdf_only = df_arr[:, 0].astype(np.float32)
            
            # Save it back as a 1D array of shape (20000,)
            np.save(df_path, sdf_only)

print("Successfully isolated SDF values in df.npy!") """


