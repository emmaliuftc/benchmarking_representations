import os
import subprocess

# 1. Set environment variables to restrict core usage

compute = input("Machine: ")
CORES = int(input("Cores: "))
START_CORE = int(input("Start core: "))

os.sched_setaffinity(0, {i for i in range(START_CORE, START_CORE + CORES)})
os.environ["NPROC"] = f"{CORES}"

if compute == "manta":
    # 2. Define your exact terminal command as a list
    command = [
        "python", "-m", "cyto_dl.train",
    #    "--config-dir", "/scratch/local/eml057/cytodl_workspace/benchmarking_representations/configs",
        "--config-dir", "/scratch/mola/eml057/cytodl_workspace/benchmarking_representations/configs",
        "experiment=npm1/pc_implicit.yaml",
        "trainer=gpu",
        "trainer.accelerator=gpu",
        "trainer.devices=1",
        "trainer.precision=32",
        # "+trainer.log_every_n_steps=10",
        # "data.batch_size=16",
        # "data.num_workers=4",
        # "data.persistent_workers=True"
    ]
elif compute == "mola":
    # # 2. Define your exact terminal command as a list
    # command = [
    #     "python", "-m", "cyto_dl.train",
    # #    "--config-dir", "/scratch/local/eml057/cytodl_workspace/benchmarking_representations/configs",
    #     # "--config-dir", "/scratch/mola/eml057/cytodl_workspace/benchmarking_representations/configs",
    #     # "experiment=pcna/pc_equiv",
    #     "experiment=other_polymorphic/pc_implicit",
    #     "+model.pretrained_ckpt_path=/scratch/mola/eml057/cytodl_workspace/benchmarking_representations/pcna/ckpts/last.ckpt",
        
    #     "trainer=cpu", # CPU
    #     "trainer.accelerator=cpu", # CPU
    #     "trainer.devices=1",
    #     "trainer.precision=32",
    #     "+trainer.log_every_n_steps=10",
    #     "data.batch_size=16",
    #     "data.num_workers=4",
    #     "data.persistent_workers=True",
    #     "data.path=/scratch/mola/eml057/cytodl_workspace/benchmarking_representations/manta_manifest.csv"
    # ]
    ...

if __name__ == "__main__":
    print(f"Starting pipeline with {CORES} cores ({START_CORE} to {START_CORE+CORES}) on {compute}.")
    
    # 3. Run the command exactly as if you typed it in the shell
    subprocess.run(command)
