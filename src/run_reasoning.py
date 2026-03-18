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
import omegaconf

from torch.nn import functional as F
from tqdm import tqdm
from copy import deepcopy
from pydantic import BaseModel
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, root_mean_squared_error, recall_score

from src.utils.dataloaders import make_breast_datasets, make_busbra, make_bus_data
from src.utils.helpers import to_one_hot, from_one_hot
from src.utils.metrics import show_confmat

from src.utils.reproducibility import set_global_seed

from src.models.reasoning import ReasoningModel
from src.models.clip import (
    CLIPRN50, 
    CLIPViT, 
    CLIPViTL, 
    BiomedCLIP,
    SigLIP
)

class ReasoningExperiment:
    def __init__(self, config):
        self.config = config

    def build_concept_model(self):
        concept_model_config = omegaconf.OmegaConf.load(self.config.concept_model.cfg)
        if concept_model_config.mode == 'clip':
            if concept_model_config.clip.backbone == 'resnet':
                return CLIPRN50(concept_model_config)
            if concept_model_config.clip.backbone == 'vit':
                return CLIPViT(concept_model_config)
            if concept_model_config.clip.backbone == 'vit-l':
                return CLIPViTL(concept_model_config)
        if concept_model_config.mode == 'siglip':
            return SigLIP(concept_model_config)
        if concept_model_config.mode == 'biomedclip':
            return BiomedCLIP(concept_model_config)

        else:
            raise Exception(f"Unknown concept model type {concept_model_config.mode}")

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
            group=self.config.wandb.group,
            name=self.config.wandb.name,
            tags=self.config.wandb.tags,
        )
        logging.info("Wandb initialized")
        logging.info("Wandb url: " + wandb.run.url)

        set_global_seed(self.config.seed)

        self.setup_data()
        self.setup_model()

    def setup_model(self):
        logging.info("Setting up model")

        self.model = ReasoningModel(
            device=self.config.device,
            use_guideline=self.config.lrm.use_guideline,
            enable_thinking=self.config.lrm.enable_thinking,
        )
        self.concept_model = self.build_concept_model()
        
        self.concept_model.load_state_dict(torch.load(
            self.config.concept_model.ckpt, 
            map_location='cpu'), 
            strict=False)
        
        self.concept_model.to(self.config.device)
        self.concept_model.eval()
        
        torch.compile(self.model)

        logging.info("Model setup complete")
        logging.info(
            f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}"
        )

    def setup_data(self):
        _, _, self.test_loader = make_bus_data(self.config)
        
        logging.info(f"Number of test batches: {len(self.test_loader)}")
        logging.info(f"Number of test samples: {len(self.test_loader.dataset)}")

    def run(self):
        self.setup()
        logging.info("Setup complete")
        logging.info("Starting evaluation")
        self.run_eval_epoch(self.test_loader, desc="test")
        logging.info("Finished running")

    @torch.no_grad()
    def run_eval_epoch(self, loader, desc="eval"):
        # setup epoch
        self.model.eval()
        
        llm_outputs = []
        for eval_iter, batch in enumerate(tqdm(loader, desc=desc)):

            # extracting relevant data from the batch
            x = batch["img_name"]
            x_img = batch["img"]
            x_img = x_img.to(self.config.device)
            y = batch["label"]
            c = batch["concepts"]

            h = self.concept_model.model.visual(x_img)
            if self.config.concept_model.name == 'CLIP':
                import joblib
                concept_probe = joblib.load('/home/harmanan/projects/aip-medilab/harmanan/breast_us/.model-weights/DDSM/clip/concept_probe_epoch12_fold0.joblib')
                c_pred = concept_probe.predict_proba(h.cpu().numpy())
                c_pred = torch.tensor(c_pred, device=self.config.device)
            else:
                c_pred = self.concept_model.get_concept_preds(h)
            
            logits = self.concept_model.get_downstream_pred(h)

            y_pred = -1
            if logits.shape[-1] == 2:
                # Binary, single logit per sample
                prob_1 = logits.sigmoid().cpu().numpy()[:, 1]
                y_pred = int((prob_1 >= 0.5).item())
            
            else:
                # Two-class or multi-class, per-class logits
                probs = logits.softmax(dim=1)                # (B, C)
                y_pred = probs.argmax(dim=1)               # (B,)
                pred_prob = probs.gather(1, y_pred.unsqueeze(1)).squeeze(1)  # (B,)
                print(pred_prob)

            c_pred = c_pred.sigmoid().cpu().numpy()[0]
            y = y.to(self.config.device)
            c = c.to(self.config.device)
            c = c.float()

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

            metadata_for_llm = {
                "birads": birads,
                "y": batch["label"].item(),
                "y_pred": y_pred,
                "concepts": c_pred,
                "concepts_gt": c.cpu().numpy()[0],
                "selected_concepts": self.config.cbm.concepts,
                "dataset": self.config.data.dataset,
                "elevation": elevation[0] if elevation else None,
                "location": location[0] if location else None,
                "species_name": species_name[0] if species_name else None,
                "use_guideline": self.config.lrm.use_guideline,
                "enable_thinking": self.config.lrm.enable_thinking,
            }
            
            # run the model
            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                # forward pass                
                text_output = self.model(metadata_for_llm)

                data = {
                    'img_name': x,
                    'birads': [birads],
                    'label': [y.cpu().item() if torch.is_tensor(y) else y],
                    'concepts': c.cpu().tolist() if torch.is_tensor(c) else c,
                    'llm_output': [text_output]
                }
                llm_outputs.append(data)

            # compute and log metrics
            epoch_metrics = {
                f"{desc}/img_name": wandb.Html(f'<p>{batch["img_name"]}</p>'),
                f"{desc}/birads": birads,
                f"{desc}/label": batch["label"],
                f"{desc}/y_pred": y_pred,
                f"{desc}/concepts": wandb.Html(f'<p>{batch["concepts"]}</p>'),
                f"{desc}/pred_concepts": wandb.Html(f'<p>{c_pred}</p>'),
                f"{desc}/llm_output": wandb.Html(f'<p>{text_output}</p>'),
                f"{desc}/prompt": wandb.Html(f'<p>{self.model.get_prompt()}</p>'),
            }

            wandb.log(epoch_metrics)

        return