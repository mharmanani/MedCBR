import logging
import os
import typing as tp
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import wandb
from einops import rearrange, repeat
from matplotlib import pyplot as plt

from torch.nn import functional as F
from tqdm import tqdm
from copy import deepcopy
from pydantic import BaseModel
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, root_mean_squared_error, recall_score

from src.utils.losses import _make_loss_function
from src.utils.dataloaders import make_breast_datasets, make_busbra, make_bus_data, named_concept_bank
from src.utils.helpers import to_one_hot, from_one_hot
from src.utils.metrics import show_confmat

from src.utils.reproducibility import set_all_rng_states, get_all_rng_states, set_global_seed

from src.models.lvlm import LVLMGenerator

def parse_text_for_concepts(summary, concept_bank, discover=False):
    """
    Parse the summary to extract the concepts.
    """
    # Remove strings that may directly give away the class names 
    concept_bank = {
        'shadowing': 0, # index of the concept in the concept bank
        'enhancement': 1,
        'halo': 2,
        'calcification': 3,
        'skin thickening': 4,
        'circumscribed': 5,
        'spiculated': 6,
        'indistinct': 7,
        'angular': 8,
        'microlobulated': 9,
        'regular': 10,
        'hyperechoic': 11,
        'hypoechoic': 12,
        'heterogeneous': 13,
        'cystic': 14
    }
        
    concepts = []
    for concept in concept_bank:
        concepts.append(0.0)
    for sentence in summary.split('.'):
        sentence = sentence.strip().lower()
        for concept in concept_bank.keys():
            if concept in sentence:
                if 'no' in sentence or 'not' in sentence:
                    pass
                if 'has' in sentence or 'presents' in sentence or 'shows' in sentence:
                    concepts[concept_bank[concept]] = 1.0

    return concepts

class LVLMExperiment:
    def __init__(self, config):
        self.config = config

    def setup(self):
        logging.basicConfig(
            level=logging.INFO if not self.config.debug else logging.DEBUG,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[logging.StreamHandler()],
        )
        logging.info("Setting up experiment")

        if self.config.debug:
            self.config.wandb.name = "debug"
        print(self.config)
        wandb.init(
            project=self.config.wandb.project,
            entity="medcbm",
            group=self.config.wandb.group,
            name=self.config.wandb.name,
            tags=self.config.wandb.tags,
        )
        logging.info("Wandb initialized")
        logging.info("Wandb url: " + wandb.run.url)

        if self.config.checkpoint_dir is not None:
            os.makedirs(self.config.checkpoint_dir, exist_ok=True)
            self.exp_state_path = os.path.join(
                self.config.checkpoint_dir, "experiment_state.pth"
            )
            if os.path.exists(self.exp_state_path):
                logging.info("Loading experiment state from experiment_state.pth")
                self.state = torch.load(self.exp_state_path)
            else:
                logging.info("No experiment state found - starting from scratch")
                self.state = None
        else:
            self.exp_state_path = None
            self.state = None

        #set_global_seed(self.config.seed)

        self.setup_data()

        self.setup_model()
        if self.state is not None:
            self.model.load_state_dict(self.state["model"])
        if self.config.pretrained:
            print(f"Loading pretrained weights from {self.config.from_ckpt}")
            checkpoint_dict = torch.load(self.config.from_ckpt, map_location='cpu')
            pretrained_weights = {}
            for param in checkpoint_dict:
                if 'clf_layer' not in param:
                    pretrained_weights[param.replace('backbone.', '')] = checkpoint_dict[param]
            
            if self.config.pretraining.style == 'supervised':
                self.model.load_state_dict(checkpoint_dict)
            elif self.config.pretraining.style == 'mae':
                self.model.model.load_state_dict(pretrained_weights)    

        self.gradient_scaler = torch.cuda.amp.GradScaler()
        if self.state is not None:
            self.gradient_scaler.load_state_dict(self.state["gradient_scaler"])

        self.epoch = 0 if self.state is None else self.state["epoch"]
        logging.info(f"Starting at epoch {self.epoch}")
        self.best_score = 0 if self.state is None else self.state["best_score"]
        logging.info(f"Best score so far: {self.best_score}")
        if self.state is not None and "rng" in self.state.keys():
            rng_state = self.state["rng"]
            set_all_rng_states(rng_state)

    def setup_model(self):
        logging.info("Setting up model")

        self.model = LVLMGenerator(device=self.config.device, prompt_mode=self.config.lvlm.prompt_mode, lvlm_name=self.config.backbone)
        self.model.to(self.config.device)
        self.model.eval()
        
        torch.compile(self.model)

        logging.info("Model setup complete")
        logging.info(
            f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}"
        )
        logging.info(
            f"Number of trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}"
        )

        # setup criterion
        self.loss_fn = _make_loss_function(self.config)

    def setup_optimizer(self):
        from torch.optim import AdamW

        class LRCalculator:
            def __init__(
                self, frozen_epochs, warmup_epochs, total_epochs, niter_per_ep
            ):
                self.frozen_epochs = frozen_epochs
                self.warmup_epochs = warmup_epochs
                self.total_epochs = total_epochs
                self.niter_per_ep = niter_per_ep

            def __call__(self, iter):
                if iter < self.frozen_epochs * self.niter_per_ep:
                    return 0
                elif (
                    iter < (self.frozen_epochs + self.warmup_epochs) * self.niter_per_ep
                ):
                    return (iter - self.frozen_epochs * self.niter_per_ep) / (
                        self.warmup_epochs * self.niter_per_ep
                    )
                else:
                    cur_iter = (
                        iter
                        - (self.frozen_epochs + self.warmup_epochs) * self.niter_per_ep
                    )
                    total_iter = (
                        self.total_epochs - self.warmup_epochs - self.frozen_epochs
                    ) * self.niter_per_ep
                    return 0.5 * (1 + np.cos(np.pi * cur_iter / total_iter))

        self.optimizer = AdamW(self.model.parameters(), weight_decay=self.config.optimizer.wd)
        from torch.optim.lr_scheduler import LambdaLR

        self.lr_scheduler = LambdaLR(
            self.optimizer,
            [
                LRCalculator(
                    self.config.optimizer.encoder_frozen_epochs,
                    self.config.optimizer.encoder_warmup_epochs,
                    self.config.training.num_epochs,
                    len(self.train_loader),
                )
            ],
        )

    def setup_data(self):
        (self.train_loader, 
         self.val_loader, 
         self.test_loader
        ) = make_bus_data(self.config)

        #print(self.train_loader.dataset.metadata.species)
        
        logging.info(f"Number of training batches: {len(self.train_loader)}")
        logging.info(f"Number of validation batches: {len(self.val_loader)}")
        logging.info(f"Number of test batches: {len(self.test_loader)}")
        logging.info(f"Number of training samples: {len(self.train_loader.dataset)}")
        logging.info(f"Number of validation samples: {len(self.val_loader.dataset)}")
        logging.info(f"Number of test samples: {len(self.test_loader.dataset)}")

    def run(self):
        self.setup()
        logging.info("Setup complete")
        
        if self.config.data.fold == 0:
            logging.info(f"Evaluating on training set")
            self.run_eval_epoch(self.train_loader, desc="train")

        if self.config.data.fold == 1:
            logging.info("Running evaluation on validation set")
            self.run_eval_epoch(self.val_loader, desc="val")

        if self.config.data.fold == 2:
            logging.info("Running evaluation on test set")
            self.run_eval_epoch(self.test_loader, desc="test")
        
        logging.info("Finished running")
        self.teardown()

    @torch.no_grad()
    def run_eval_epoch(self, loader, desc="eval"):
        # setup epoch
        self.model.eval()

        # set of comma-separated strings
        llm_outputs = []

        for eval_iter, batch in enumerate(tqdm(loader, desc=desc)):

            # extracting relevant data from the batch
            x = batch["img_name"]
            mask = batch["mask_name"]
            
            y = batch["label"]
            c = batch["concepts"]

            # breast cancer features
            if "birads" in batch.keys(): 
                birads = batch["birads"].item()
            else: birads = None

            # skin cancer features
            if "elevation" in batch.keys():
                elevation = batch["elevation"]
            else: elevation = None

            if "location" in batch.keys():
                location = batch["location"]
            else: location = None
            
            # cub features
            if "species_name" in batch.keys():
                species_name = batch["species_name"]
            else: species_name = None

            y = y.to(self.config.device)
            c = c.to(self.config.device)
            c = c.float()

            metadata_for_lvlm = {
                "birads": birads,
                "y": batch["label"].item(),
                "concepts": c.tolist()[0],
                "selected_concepts": self.config.cbm.concepts,
                "dataset": self.config.data.dataset,
                "elevation": elevation[0] if elevation else None,
                "location": location[0] if location else None,
                "species_name": species_name[0] if species_name else None,
            }

            if "roi_id" in batch.keys():
                metadata_for_lvlm["roi_id"] = batch["roi_id"][0]

            if self.config.lvlm.include_mask:
                metadata_for_lvlm["mask"] = mask
            if self.config.lvlm.include_bbox:
                pass

            # run the model
            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                # forward pass
                text_output = self.model.generate_clinical_report(x, metadata_for_lvlm)
                
                if self.config.backbone != 'medgemma':
                    text_output = text_output[0]
                
                data = {
                    'img_name': x,
                    'label': [y.item()],
                    'concepts': c.tolist(),
                    'llm_output': [text_output],
                }
                llm_outputs.append(data)
            
            # log metrics
            step_metrics = {}
            
            main_lr = self.config.optimizer.main_lr
            step_metrics["main_lr"] = main_lr

            if "birads" in batch.keys():
                birads = batch["birads"],
            else:
                birads = "N/A"

            # compute and log metrics
            epoch_metrics = {
                f"{desc}/img_name": wandb.Html(f'<p>{batch["img_name"]}</p>'),
                f"{desc}/birads": birads,
                f"{desc}/label": batch["label"],
                f"{desc}/concepts": wandb.Html(f'<p>{batch["concepts"]}</p>'),
                f"{desc}/llm_output": wandb.Html(f'<p>{text_output}</p>'),
            }

            wandb.log(epoch_metrics)

        wandb_table = wandb.Table(
            dataframe=llm_out_df,
            columns=col_names.split(','),
        )
        wandb.log({f"Tables/LLM_Outputs_{desc}": wandb_table})

        return epoch_metrics

    def save_experiment_state(self):
        if self.exp_state_path is None:
            return
        logging.info(f"Saving experiment snapshot to {self.exp_state_path}")
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": self.epoch,
                "best_score": self.best_score,
                "gradient_scaler": self.gradient_scaler.state_dict(),
                "rng": get_all_rng_states(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
            },
            self.exp_state_path,
        )

    def save_model_weights(self, score, is_best_score=False):
        if self.config.checkpoint_dir is None:
            return

        if not is_best_score:
            fname = f"model_epoch{self.epoch}.ckpt"
        else:
            fname = "best_model.ckpt"

        logging.info("Saving model to checkpoint directory")
        logging.info(f"Checkpoint directory: {self.config.checkpoint_dir}")
        torch.save(
            self.model.state_dict(),
            os.path.join(self.config.checkpoint_dir, fname),
        )

    def teardown(self):
        # remove experiment state file
        if self.exp_state_path is not None:
            os.remove(self.exp_state_path)


