import os
import torch
import numpy as np
from skimage.measure import marching_cubes
import trimesh
from omegaconf import OmegaConf
from hydra.utils import instantiate

def main():
    # --- 1. Configuration ---
    config_path = "/scratch/mola/eml057/cytodl_workspace/benchmarking_representations/logs/train/runs/npm1_variance/catchall_run/2026-06-24_12-28-37/.hydra/config.yaml"
    checkpoint_path = "/scratch/mola/eml057/cytodl_workspace/benchmarking_representations/npm1_variance/ckpts/epoch_248.ckpt"
    pc_path = "/scratch/mola/eml057/cytodl_workspace/data/shapenet_format/00000000/001/pointcloud.npz"
    output_filename = "reconstructed_npm1.stl"

    # --- 2. Load Model via Hydra ---
    print("Rebuilding architecture directly from Hydra config...")
    cfg = OmegaConf.load(config_path)
    model = instantiate(cfg.model, _recursive_=False)
    
    print(f"Loading weights from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    model.cuda()

    # --- 3. Load Sample Point Cloud ---
    print(f"Loading input point cloud: {pc_path}")
    data = np.load(pc_path)
    pc = data['points'][:8192].astype(np.float32) 
    pc_tensor = torch.tensor(pc).unsqueeze(0).cuda()

    # --- 4. Generate Dense 3D Query Grid ---
    print("Generating dense 3D query grid...")
    resolution = 256
    grid_bound = 250 
    
    axis = np.linspace(-grid_bound, grid_bound, resolution)
    X, Y, Z = np.meshgrid(axis, axis, axis, indexing='ij')
    
    query_points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float32)

    # --- 5. Model Inference (in chunks) ---
    print("Predicting the SDF scalar field (chunking to prevent GPU OOM)...")
    chunk_size = 20000 
    sdf_preds = []
    
    with torch.no_grad():
        for i in range(0, len(query_points), chunk_size):
            chunk = query_points[i : i + chunk_size]
            query_tensor = torch.tensor(chunk).unsqueeze(0).cuda()
            
            batch = {
                "pcloud": pc_tensor, 
                "points": query_tensor
            }
            
            preds = model(batch, decode=True, inference=True)
            
            # Unpack tuple if needed
            if isinstance(preds, tuple):
                preds = preds[0]
                
            # --- THE DICTIONARY SEARCH ---
            target_dim = chunk.shape[0] # 20000 (or 2144 for the last chunk)
            sdf_chunk_tensor = None
            
            if isinstance(preds, torch.Tensor):
                sdf_chunk_tensor = preds
            elif isinstance(preds, dict):
                for k, v in preds.items():
                    if isinstance(v, torch.Tensor) and target_dim in v.shape:
                        sdf_chunk_tensor = v
                        break
                if sdf_chunk_tensor is None:
                    sdf_chunk_tensor = max([v for v in preds.values() if isinstance(v, torch.Tensor)], key=lambda x: x.numel())
                    
            if sdf_chunk_tensor is None:
                raise ValueError("Could not find a valid tensor in model output!")
                
            # --- THE MATH-BASED EXTRACTION ---
            # Count the absolute number of elements in the tensor
            total_elements = sdf_chunk_tensor.numel()
            
            if total_elements == target_dim * 4:
                # The model output 4 channels (SDF + XYZ). Reshape and grab column 0.
                flat_chunk = sdf_chunk_tensor.view(target_dim, 4)[:, 0].cpu().numpy()
            elif total_elements == target_dim:
                # The model perfectly output 1 channel.
                flat_chunk = sdf_chunk_tensor.flatten().cpu().numpy()
            else:
                raise ValueError(f"Unexpected tensor! Expected {target_dim} points, but tensor has {total_elements} elements. Shape: {sdf_chunk_tensor.shape}")
                
            sdf_preds.append(flat_chunk)

    # --- 6. Marching Cubes Extraction ---
    print("Running Marching Cubes algorithm...")
    sdf_pred = np.concatenate(sdf_preds)
    
    # Final sanity check before reshaping
    if len(sdf_pred) != resolution**3:
        raise ValueError(f"Total points ({len(sdf_pred)}) does not match grid size ({resolution**3})!")
        
    sdf_volume = sdf_pred.reshape(resolution, resolution, resolution)
    
    try:
        vertices, faces, normals, values = marching_cubes(sdf_volume, level=0.0)
        
        voxel_size = (2.0 * grid_bound) / (resolution - 1)
        vertices = vertices * voxel_size - grid_bound
        
        # --- 7. Save as STL ---
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.export(output_filename)
        print(f"\nSuccess! 3D mesh saved to: {os.path.abspath(output_filename)}")
        
    except ValueError:
        print("\nMarching Cubes failed.")
        print(f"Predicted SDF Range: Min {sdf_volume.min():.4f}, Max {sdf_volume.max():.4f}")

if __name__ == "__main__":
    main()