import numpy as np
import os
import pandas as pd
import glob
import random
import warnings
import trimesh
import point-cloud-utils as pcu

data_dir = "/scratch/mola/eml057/cytodl_workspace/data/stl_scaled/"

files = glob.glob(os.path.join(data_dir, "*.stl"))
files.sort()

v, f, n = pcu.load_mesh_vfn

# in progres 