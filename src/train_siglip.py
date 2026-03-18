
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

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier


from src.utils.dataloaders import make_bus_data, make_report_from_concepts
from src.utils.metrics import compute_conceptwise_metrics, compute_classification_metrics, compute_multiclass_metrics
from src.utils.losses import concept_unweighted_ce_loss, concept_weighted_ce_loss

from src.utils.reproducibility import set_all_rng_states, get_all_rng_states, set_global_seed
from src.models.clip import SigLIP

class SigLIPExperiment:
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
        wandb.init(
            project=self.config.wandb.project,
            group=self.config.wandb.group,
            name=self.config.wandb.name,
            tags=self.config.wandb.tags,
        )
        logging.info("Wandb initialized")
        logging.info("Wandb url: " + wandb.run.url)

        if self.config.checkpoint_dir is not None:
            if not os.path.exists(self.config.checkpoint_dir):
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

        set_global_seed(self.config.seed)

        self.setup_data()

        self.setup_model()
        if self.state is not None:
            self.model.load_state_dict(self.state["model"])

        self.setup_optimizer()
        if self.state is not None:
            self.optimizer.load_state_dict(self.state["optimizer"])
            self.lr_scheduler.load_state_dict(self.state["lr_scheduler"])

        self.gradient_scaler = torch.cuda.amp.GradScaler()
        if self.state is not None:
            self.gradient_scaler.load_state_dict(self.state["gradient_scaler"])

        self.epoch = 0 if self.state is None else self.state["epoch"]
        logging.info(f"Starting at epoch {self.epoch}")
        self.best_score = 0 if self.state is None else self.state["best_score"]
        logging.info(f"Best score so far: {self.best_score}")
        if self.state is not None:
            rng_state = self.state["rng"]
            set_all_rng_states(rng_state)

    def setup_model(self):
        logging.info("Setting up model")

        self.model = SigLIP(self.config)

        if self.config.pretrained and self.config.from_ckpt is not None:
            logging.info(
                f"Loading pretraining weights from {self.config.from_ckpt}"
            )
            self.model.load_state_dict(
                torch.load(self.config.from_ckpt, map_location="cpu")
            )

        self.model.to(self.config.device)
        torch.compile(self.model)        

        # downstream classifier
        self.det_probe = RandomForestClassifier(
            n_estimators=100,
            random_state=self.config.seed,
            class_weight='balanced',
            n_jobs=-1,
            verbose=1
        )
        
        # concept detection classifier
        self.concept_probe = OneVsRestClassifier(
            RandomForestClassifier(
                n_estimators=100,
                random_state=self.config.seed,
                class_weight='balanced',
                n_jobs=-1,
                verbose=1
            ),
            n_jobs=-1
        )

        self.det_loss_fn = F.cross_entropy
        self.concept_loss_fn = concept_weighted_ce_loss

        logging.info("Model setup complete")
        logging.info(
            f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}"
        )
        logging.info(
            f"Number of trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}"
        )

    def setup_optimizer(self):
        from torch.optim import AdamW, SGD

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

        self.optimizer = AdamW(self.model.parameters(),
                               lr=self.config.optimizer.lr, 
                               weight_decay=self.config.optimizer.wd)
        
        from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
        if self.config.training.lr_scheduler == "cosine":
            self.lr_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs * len(self.train_loader),
                eta_min=1e-6,
            )
        
        elif self.config.training.lr_scheduler == "warmup":
            # warmup from 1e-7 to 1e-5 and then stay at 1e-5 for the rest of the training
            logging.info("Using warmup scheduler")
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lambda iter: (
                    1e-7 + (1e-5 - 1e-7) * 
                    (iter / (self.config.optimizer.warmup_epochs * len(self.train_loader)))
                ) if iter < self.config.optimizer.warmup_epochs * len(self.train_loader) else 1e-5
            )
        
        elif self.config.training.lr_scheduler == "cosine_warmup":
            logging.info("Using cosine warmup scheduler")
            self.lr_scheduler = LambdaLR(
                self.optimizer,
                [
                    LRCalculator(
                        self.config.optimizer.frozen_epochs,
                        self.config.optimizer.warmup_epochs,
                        self.config.training.num_epochs,
                        len(self.train_loader),
                    )
                ],
            )

    def setup_data(self):
        (
            self.train_loader, 
            self.val_loader, 
            self.test_loader 
        ) = make_bus_data(self.config)

        print("Train loader:", self.train_loader)
        print("Validation loader:", self.val_loader)
        print("Test loader:", self.test_loader)
        
        
        logging.info(f"Number of training batches: {len(self.train_loader)}")
        logging.info(f"Number of validation batches: {len(self.val_loader)}")
        logging.info(f"Number of test batches: {len(self.test_loader)}")
        logging.info(f"Number of training samples: {len(self.train_loader.dataset)}")
        logging.info(f"Number of validation samples: {len(self.val_loader.dataset)}")
        logging.info(f"Number of test samples: {len(self.test_loader.dataset)}")

    def run(self):
        self.setup()
        for self.epoch in range(self.epoch, self.config.training.num_epochs):
            logging.info(f"Epoch {self.epoch}")
            
            try:
                self.save_experiment_state()
            except Exception as e:
                logging.error(f"Error saving experiment state: {e}")

            self.run_train_epoch(self.train_loader, desc="train")

            val_metrics = self.run_eval_epoch(self.val_loader, desc="val")

            if val_metrics is not None:
                tracked_metric = val_metrics["val/loss"]
                new_record = tracked_metric > self.best_score
            else:
                new_record = None

            if new_record:
                self.best_score = tracked_metric
                logging.info(f"New best score: {self.best_score}")
            
            metrics = self.run_eval_epoch(self.test_loader, desc="test")
            test_score = metrics["test/concept_auc"]          
            self.save_model_weights(score=test_score, is_best_score=new_record)  

        logging.info("Finished training")
        self.teardown()

    def run_train_epoch(self, loader, desc="train"):
        # setup epoch
        self.model.train()
        self.model.classifier.train()
        for c in range(len(self.config.cbm.concepts)):
            adapter = self.model.concept_classifiers[c]
            adapter.train()

        loss_per_step = np.array([])

        all_img_feats = []
        all_txt_feats = []
        all_labels = []
        all_concepts = []

        for train_iter, batch in enumerate(tqdm(loader, desc=desc)):
            print(f'batch: {batch}')
            # extracting relevant data from the batch
            x = batch["img_name"]
            t = batch["llm_output"]
            y = batch["label"]
            c = batch["concepts"]

            print(y.shape)
            
            #x = x.to(self.config.device)
            y = y.to(self.config.device)
            c = c.to(self.config.device)
            c = c.float()

            B = len(x)

            # run the model
            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                # forward pass                       
                skip_batch = False            
                for i in range(len(t)):
                    if str(t[i]) == "nan":
                        print(t[i], '\n')
                        skip_batch = True
                        break
                if skip_batch:
                    continue

                print(t)
                
                img_feats, txt_feats, _ = self.model(x, t)
                clip_loss = self.model.apply_clip_loss(img_feats, txt_feats)

                y_pred = self.model.get_downstream_pred(img_feats)
                c_pred = self.model.get_concept_preds(img_feats)

                det_loss = self.det_loss_fn(
                    y_pred, y
                )  # y is the label tensor, y_pred is the logits tensor
                
                concept_loss = self.concept_loss_fn(
                    c_pred, c, concept_weights=self.config.cbm.concept_weights
                )  # c is the concepts tensor, c_pred is the logits tensor

                loss = (
                    self.config.clip.clip_weight * clip_loss
                    + self.config.clip.det_weight * det_loss
                    + self.config.clip.concept_weight * concept_loss
                )

                # backward pass
                if self.config.use_amp:
                    self.gradient_scaler.scale(loss).backward()
                    self.gradient_scaler.step(self.optimizer)
                    self.gradient_scaler.update()
                    self.optimizer.zero_grad()
                else:
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if self.config.training.lr_scheduler:
                    self.lr_scheduler.step()

                for j in range(B):
                    all_img_feats.append(img_feats[j, :].detach().cpu().numpy())
                    all_txt_feats.append(txt_feats[j, :].detach().cpu().numpy())
                    all_labels.append(y[j].detach().cpu().numpy())
                    all_concepts.append(c[j, :].detach().cpu().numpy())

            loss_per_step = np.concatenate([loss_per_step, loss.detach().cpu().item()], axis=None)
            
            # log metrics
            step_metrics = {
                "loss_per_step": loss.item(),
            }

            wandb.log(step_metrics)

        epoch_metrics = {
            "epoch": self.epoch,
            "train/loss": loss_per_step.mean(),
        }

        wandb.log(epoch_metrics)


        self.det_probe.fit(X=all_img_feats, y=all_labels)
        self.concept_probe.fit(X=all_img_feats, y=all_concepts)

        return epoch_metrics

    @torch.no_grad()
    def run_eval_epoch(self, loader, desc="eval"):
        self.model.eval()
        self.model.classifier.eval()
        for c in range(len(self.config.cbm.concepts)):
            adapter = self.model.concept_classifiers[c]
            adapter.eval()

        all_img_feats = []
        all_txt_feats = []
        all_labels = []
        all_concepts = []

        for eval_iter, batch in enumerate(tqdm(loader, desc=desc)):
            # extracting relevant data from the batch
            x = batch["img_name"]
            t = batch["llm_output"]
            y = batch["label"]
            c = batch["concepts"]
            
            #x = x.to(self.config.device)
            y = y.to(self.config.device)
            c = c.to(self.config.device)
            c = c.float()

            B = len(x)

            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                # forward pass  
                skip_batch = False            
                for i in range(len(t)):
                    if str(t[i]) == "nan":
                        print(t[i], '\n')
                        skip_batch = True
                        break
                if skip_batch:
                    continue
                
                img_feats, txt_feats, _ = self.model(x, t)
                loss = self.model.apply_clip_loss(img_feats, txt_feats)

                for j in range(B):
                    all_img_feats.append(img_feats[j, :].detach().cpu().numpy())
                    all_txt_feats.append(txt_feats[j, :].detach().cpu().numpy())
                    all_labels.append(y[j].detach().cpu().numpy())
                    all_concepts.append(c[j, :].detach().cpu().numpy())

        
        epoch_metrics = {
            "epoch": self.epoch,
            f"{desc}/loss": loss.item(),
        } 
        
        if self.config.data.dataset != 'BUSBRA':
            y_pred_probe = self.det_probe.predict_proba(all_img_feats)
            c_pred_probe = self.concept_probe.predict_proba(all_img_feats)
            clf_metrics_probe = compute_classification_metrics(all_labels, 
                                                               y_pred_probe,
                                                               multi_class=self.config.data.dataset == "CUB",
                                                               desc=desc)
            concept_metrics_probe = compute_conceptwise_metrics(all_concepts, 
                                                        c_pred_probe, 
                                                        selected_concepts=self.config.cbm.concepts,
                                                        dataset=self.config.data.dataset,
                                                        desc=desc)

            for metric, value in concept_metrics_probe.items():
                epoch_metrics[f"{desc}/{metric}_probe"] = value
            for metric, value in clf_metrics_probe.items():
                epoch_metrics[f"{desc}/{metric}_probe"] = value
    
        y_pred = self.model.get_downstream_pred(torch.tensor(all_img_feats, device=self.config.device))
        c_pred = self.model.get_concept_preds(torch.tensor(all_img_feats, device=self.config.device))
        y_pred = y_pred.detach().cpu().numpy()
        c_pred = c_pred.detach().cpu().numpy()

        print(f"y_pred shape: {y_pred.shape}, c_pred shape: {c_pred.shape}")
        print(f"all_labels shape: {np.array(all_labels).shape}, all_concepts shape: {np.array(all_concepts).shape}")
        
        clf_metrics = compute_classification_metrics(
            all_labels, y_pred, desc=desc, multi_class=self.config.data.dataset=="CUB"
        )
        concept_metrics = compute_conceptwise_metrics(
            all_concepts, 
            c_pred, 
            selected_concepts=self.config.cbm.concepts,
            dataset=self.config.data.dataset,
            desc=desc
        )

        for metric, value in concept_metrics.items():
            epoch_metrics[f"{desc}/{metric}"] = value
        for metric, value in clf_metrics.items():
            epoch_metrics[f"{desc}/{metric}"] = value

        wandb.log(epoch_metrics)

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



