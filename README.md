# Benchmarking Representations

Code for training and benchmarking morphology appropriate representation learning methods, associated with the following manuscript.

> **Interpretable representation learning for 3D multi-piece intracellular structures using point clouds**
>
> Ritvik Vasan, Alexandra J. Ferrante, Antoine Borensztejn, Christopher L. Frick, Philip Garrison, Nathalie Gaudreault, Saurabh S. Mogre, Fatwir S. Mohammed, Benjamin Morris, Guilherme G. Pires, Daniel Saelid, Susanne M. Rafelski, Julie A. Theriot, Matheus P. Viana
>
> https://www.nature.com/articles/s41592-025-02729-9

Our analysis is organized as follows.

1. Single cell images
2. Preprocessing (result: pointclouds and SDFs)
   1. Punctate structures
      1. Alignment, masking, and registration
      2. Generate pointclouds
   2. Polymorphic structures: Generate SDFs
3. Model training (result: checkpoint)
4. Model inference (results: embeddings, model cost statistics)
5. Interpretability analysis (results: figures)

Continue below for guidance on using these models on your own data.
If you'd like to reproduce this analysis on our data, check out the following documentation.

- [Main usage documentation](./docs/USAGE.md) for reproducing the figures in the paper from published pointclouds and SDFs, including model training and inference (steps 3-5).
- [Preprocessing documentation](./docs/PREPROCESSING.md) for generating pointclouds and SDFs from from our input movies (step 2).
- [Development documentation](./docs/DEVELOPMENT.md) for guidance working on the code in this repository.

# Quickstart

Use the cytodl api to run any given experiment. For e.g., train a rotation invariant point cloud autoencoder on the PCNA dataset

```bash
from cyto_dl.api import CytoDLModel
from pathlib import Path

model = CytoDLModel()
model.root= Path(os.getcwd())
model.load_default_experiment('pcna/pc_equiv', output_dir = './')

model.print_config()

model.train()
```

# Contact

Allen Institute for Cell Science (cells@alleninstitute.org)

# XYZ Point Cloud Setup on macOS

Certain binary segmented XYZ datasets lack the intrinsic scalar feature channel; running the `pc_equiv` experiment on macOS requires a specific patched environment to prevent tensor shape mismatches and CPU segmentation faults (among other things).

## 1. Library Prerequisite
You should install the matching patched version of the `cyto-dl` core library in its feature branch:

```bash
pip install git+https://github.com/emmaliuftc/cyto-dl.git@fix-macos-xyz-training
```

## 2. Execution Command

```bash
export MKL_DEBUG=1 \
       MKL_NUM_THREADS=1 \
       OMP_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 \
       OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

python -X faulthandler -m cyto_dl.train \
    --config-dir $(pwd)/configs \
    experiment=pcna/pc_equiv \
    trainer=cpu \
    trainer.accelerator=cpu \
    trainer.devices=1 \
    trainer.log_every_n_steps=1 \
    data.batch_size=1 \
    data.num_workers=0 \
    data.persistent_workers=False \
    data.multiprocessing_context=null \
    'model.reconstruction_loss={pcloud: {_target_: cyto_dl.nn.losses.GeomLoss, p: 1, blur: 0.01}}'
```

Flag explanations:
- `MKL/OMP/VECLIB environment variables`: Disables parallelized CPU matrix math blocks to prevent macOS architecture segmentation faults during graph calculations.
- `trainer.log_every_n_steps=1`: Forces PyTorch Lightning to log evaluation metrics on every single step rather than defaulting to 50, allowing real-time tracking for small micro-batches.
- `data.batch_size=1`: Matches single-sample tensor allocations for steady training steps.
- `data.num_workers=0`: Forces the dataset to load directly on the main execution thread, bypassing the unsafe process-forking behaviors native to Python on macOS.