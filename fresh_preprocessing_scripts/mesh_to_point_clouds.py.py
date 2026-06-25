import numpy as np
import os
import glob
import trimesh
import point_cloud_utils as pcu

data_dir = "/scratch/mola/eml057/cytodl_workspace/data/stl_scaled/"

files = glob.glob(os.path.join(data_dir, "*.stl"))
files.sort()

# Define sample sizes

# NUM_SURFACE_POINTS = 8192
# NUM_SDF_POINTS = 20000

NUM_SURFACE_POINTS = 16384
NUM_SDF_POINTS = 40000


for f in files:
    file_id = os.path.basename(f).split('.')[0]
    # 1. Load the mesh
    v, f, n = pcu.load_mesh_vfn(f)
    
    # 2. Make the mesh watertight
    resolution = 10_000
    vw, fw = pcu.make_mesh_watertight(v, f, resolution)
    
    # ==========================================
    # 3. Surface Point Cloud Sampling
    # ==========================================
   # Sample random points on the surface (returns face indices and barycentric coords)
    f_i, bc = pcu.sample_mesh_random(vw, fw, num_samples=NUM_SURFACE_POINTS)
    
    # Interpolate to get the actual (8192, 3) XYZ coordinates
    surface_points = pcu.interpolate_barycentric_coords(fw, f_i, bc, vw)
    
    # Verify the shape is exactly (8192, 3)
    assert surface_points.shape == (NUM_SURFACE_POINTS, 3), f"Expected {(NUM_SURFACE_POINTS, 3)}, got {surface_points.shape}"
    
    # Optional: Apply Lloyd relaxation if you strictly need a more uniform (blue-noise-like) distribution
    # surface_points = pcu.lloyd_relax_point_cloud(surface_points)
    
    # ==========================================
    # 4. SDF Volume Point Cloud Sampling
    # ==========================================
    # Calculate bounding box of the watertight mesh
    bbox_min = np.min(vw, axis=0)
    bbox_max = np.max(vw, axis=0)
    
    # Add a 10% margin to the bounding box to capture the exterior SDF gradient
    margin = 0.1 * (bbox_max - bbox_min)
    bbox_min -= margin
    bbox_max += margin
    
    # Generate 20,000 random query points within the padded bounding box
    sdf_query_points = np.random.uniform(low=bbox_min, high=bbox_max, size=(NUM_SDF_POINTS, 3))
    
    # Compute the signed distance from the query points to the watertight mesh
    # pcu.signed_distance_to_mesh returns: distances, closest_points, and face_indices
    sdf_distances, _, _ = pcu.signed_distance_to_mesh(sdf_query_points, vw, fw)
    
    # Create the 4D point cloud (SDF value + XYZ coordinates)
    # Reshape distances to be a column vector and concatenate with XYZ
    sdf_4d_cloud = np.column_stack((sdf_distances, sdf_query_points))
    
    # ==========================================
    # 5. Save the Outputs (Example)
    # ==========================================    
    
    # Save as .npy files for fast loading during training
    np.save(os.path.join("/scratch/mola/eml057/cytodl_workspace/data/point_clouds", f"{file_id}_surface_{NUM_SURFACE_POINTS}.npy"), surface_points)
    np.save(os.path.join("/scratch/mola/eml057/cytodl_workspace/data/point_clouds", f"{file_id}_sdf_{NUM_SDF_POINTS}.npy"), sdf_4d_cloud)
    
    print(f"Processed {file_id}: Surface Shape {surface_points.shape}, SDF Shape {sdf_4d_cloud.shape}")