import argparse
import omegaconf
import rich
import rich.pretty
import datetime
import submitit
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from src.utils.reproducibility import set_global_seed

from src.train_cbm import CBMExperiment
from src.run_lvlm import LVLMExperiment
from src.run_reasoning import ReasoningExperiment

# CLIP Models
from src.train_siglip import SigLIPExperiment
from src.train_clip import CLIPExperiment
from src.train_biomedclip import BiomedCLIPExperiment

PROJ_DIR = "~/projects/aip-medilab/harmanan/breast_us"

class Main:
    def __init__(self, conf):
        self.args = conf

    def __call__(self):
        SLURM_JOB_ID = os.getenv("SLURM_JOB_ID")
        os.environ["TQDM_MININTERVAL"] = "30"
        os.environ["WANDB_RUN_ID"] = f"{SLURM_JOB_ID}"
        os.environ["WANDB_RESUME"] = "allow"
        CKPT_DIR = f'/scratch/harmanan/ckpt'

        conf.slurm_job_id = SLURM_JOB_ID

        # ResNet baseline
        if conf.mode == "cbm" or conf.mode == "cbm-base":
            experiment = CBMExperiment(conf)
        
        # CLIP models
        elif conf.mode == "clip":
            experiment = CLIPExperiment(conf)
        elif conf.mode == "siglip":
            experiment = SigLIPExperiment(conf)
        elif conf.mode == "biomedclip":
            experiment = BiomedCLIPExperiment(conf)
        
        # Large Language Models
        elif conf.mode == 'llama':
            experiment = LVLMExperiment(conf)
        elif conf.mode == 'reasoning':
            experiment = ReasoningExperiment(conf)
        
        experiment.run()

    def checkpoint(self):
        return submitit.helpers.DelayedSubmission(Main(self.args))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MedCBM')
    
    parser.add_argument('-y', '--yaml', type=str, default='cbm')
    parser.add_argument('-o', '--overrides', nargs="+", default=[])
    
    args = parser.parse_args()
    conf = omegaconf.OmegaConf.load(f"config/{args.yaml}.yaml")

    if conf.seed: # If seed is set, set the global seed
        set_global_seed(conf.seed)

    # Override config with command line arguments
    conf = omegaconf.OmegaConf.merge(conf, omegaconf.OmegaConf.from_dotlist(args.overrides))

    # Save config
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    # Set W&B run name and checkpoint directory
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    conf.wandb.name = f'{conf.exp_name}_fold={conf.data.fold}_{timestamp}'        
    if not conf.save_weights:
        conf.checkpoint_dir = None # checkpointing takes too much space

    if conf.debug:
        conf.data.batch_size = 1
        conf.device = 'cpu'
        conf.checkpoint_dir = None
        conf.wandb.name = 'debug'

    # Run the experiment directly
    main = Main(conf)
    main()