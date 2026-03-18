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
from src.utils.dataloaders import make_bus_data
from src.utils.helpers import to_one_hot, from_one_hot
from src.utils.metrics import show_confmat, compute_conceptwise_metrics, compute_classification_metrics

from src.utils.reproducibility import set_all_rng_states, get_all_rng_states, set_global_seed

from src.models.cbm import BaseCBM, FusionCBM, TwoTrunkCBM, ProbabilisticCBM
from src.models.cmh import BaseCMH

class CBMExperiment:
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
            group=self.config.wandb.group,
            name=self.config.wandb.name,
            tags=self.config.wandb.tags,
        )
        logging.info("Wandb initialized")
        logging.info("Wandb url: " + wandb.run.url)

        if self.config.checkpoint_dir is not None:
            try:
                os.makedirs(self.config.checkpoint_dir, exist_ok=True)
            except Exception as e:
                logging.error(f"Error creating checkpoint directory: {e}")
                raise e
            
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
        if self.config.pretrained:
            if self.config.pretraining.style == "cbm":
                self.model.load_state_dict(torch.load(self.config.from_ckpt))
                # Freeze the concept layer
                # self.model.concept_layer.requires_grad_(False)
            else:
                self.model.load_backbone_weights(self.config.from_ckpt) 

        self.setup_optimizer()
        if self.state is not None:
            self.optimizer.load_state_dict(self.state["optimizer"])

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

        
        self.model = BaseCMH(
            num_concepts=self.config.cbm.num_concepts,
            num_classes=self.config.cbm.num_classes,
            backbone=self.config.cbm.backbone
        )

        self.model.to(self.config.device)
        self.model = torch.compile(self.model)

        logging.info("Model setup complete")
        logging.info(
            f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}"
        )
        logging.info(
            f"Number of trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}"
        )

        # setup criterion
        self.loss_fn = F.cross_entropy

        logging.info("Loss function setup complete")

    def setup_optimizer(self):
        logging.info("Setting up optimizer and lr scheduler")

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.optimizer.main_lr, weight_decay=self.config.optimizer.wd
        )

        logging.info("Optimizer and lr scheduler setup complete")

    def setup_data(self):
        (
            self.train_loader,
            self.val_loader,
            self.test_loader,
        ) = make_bus_data(self.config)

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
            self.save_experiment_state()

            self.run_train_epoch(self.train_loader, desc="train")

            val_metrics = self.run_eval_epoch(self.val_loader, desc="val")

            if val_metrics is not None:
                tracked_metric = val_metrics["val/auc"] if self.config.data.dataset != "CUB" else val_metrics["val/accuracy"]
                new_record = tracked_metric < self.best_score
            else:
                new_record = None

            if new_record:
                self.best_score = tracked_metric
                logging.info(f"New best score: {self.best_score}")

            metrics = self.run_eval_epoch(self.test_loader, desc="test")
            test_score = metrics["test/auc"]

            self.save_model_weights(score=test_score, is_best_score=new_record)

        logging.info("Finished training")
        self.teardown()

    def run_train_epoch(self, loader, desc="train"):
        # setup epoch
        self.model.train()

        cancer_pred, cancer_true = [], []
        concepts_pred, concepts_true = [], []
        loss_per_step = np.array([])

        for train_iter, batch in enumerate(tqdm(loader, desc=desc)):

            # extracting relevant data from the batch
            x = batch["img"]
            y = batch["label"]
            c = batch["concepts"]
            
            x = x.to(self.config.device)
            y = y.to(self.config.device)
            c = c.to(self.config.device)
            c = c.float()

            c_pred, y_pred = self.model(x)

            # loss calculation
            loss = self.loss_fn(y_pred, y)

            if self.config.use_amp:
                self.gradient_scaler.scale(loss).backward()
                self.gradient_scaler.step(self.optimizer)
                self.gradient_scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            self.optimizer.zero_grad(set_to_none=True)

            cancer_pred.append(y_pred.cpu().detach().numpy())
            cancer_true.append(y.cpu().unsqueeze(-1).detach().numpy())
            concepts_pred.append(c_pred.cpu().detach().numpy())
            concepts_true.append(c.cpu().detach().numpy())

            loss_per_step = np.concatenate([loss_per_step, loss.detach().cpu().item()], axis=None)
            
            # log metrics
            step_metrics = {}
            
            main_lr = self.config.optimizer.main_lr
            step_metrics["main_lr"] = main_lr

            wandb.log(step_metrics)

        cancer_pred = np.vstack(cancer_pred)
        cancer_true = np.vstack(cancer_true)
        cancer_true = np.vstack(cancer_true) # this is not a bug; needs to be stacked twice
        cancer_prob = cancer_pred.copy()
        cancer_pred = np.argmax(cancer_pred, axis=1)
        
        concepts_pred = np.vstack(concepts_pred)
        concepts_pred = np.where(concepts_pred >= 0.5, 1, 0)
        concepts_true = np.vstack(concepts_true)
        
        # compute and log metrics
        concept_metrics = compute_conceptwise_metrics(concepts_true, 
                                                      concepts_pred, 
                                                      selected_concepts=self.config.cbm.concepts,
                                                      dataset=self.config.data.dataset,
                                                      desc=desc)
        perf_metrics = compute_classification_metrics(cancer_true, 
                                                      cancer_prob, 
                                                      multi_class=self.config.cbm.num_classes > 2,
                                                      tune_threshold=self.config.metrics.tune_threshold)
        
        epoch_metrics = {
            "epoch": self.epoch,
            "train/loss": np.mean(loss_per_step)
        }

        for (metric_name, metric_value) in concept_metrics.items():
            epoch_metrics[f"train/{metric_name}"] = metric_value
        for (metric_name, metric_value) in perf_metrics.items():
            epoch_metrics[f"train/{metric_name}"] = metric_value

        wandb.log(epoch_metrics)
        return epoch_metrics

    @torch.no_grad()
    def run_eval_epoch(self, loader, desc="eval"):
        # setup epoch
        self.model.eval()

        cancer_pred, cancer_true = [], []
        concepts_pred, concepts_true = [], []
        loss_per_step = np.array([])

        for eval_iter, batch in enumerate(tqdm(loader, desc=desc)):

            # extracting relevant data from the batch
            x = batch["img"]
            y = batch["label"]
            c = batch["concepts"]
            
            x = x.to(self.config.device)
            y = y.to(self.config.device)
            c = c.to(self.config.device)
            c = c.float()

            # run the model
            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                # forward pass
                c_pred, y_pred = self.model(x)

                # loss calculation
                if self.config.loss == 'weighted_multi_task_ce_loss':
                    loss = self.loss_fn(y_pred, y, 
                                        c_pred, c, 
                                        gamma=self.config.cbm.gamma,
                                        concept_weights=self.config.cbm.concept_weights, 
                                        device=self.config.device)
                if self.config.loss == 'unweighted_multi_task_ce_loss':
                    loss = self.loss_fn(y_pred, y, 
                                        c_pred, c, 
                                        gamma=self.config.cbm.gamma,
                                        device=self.config.device)
                if self.config.loss == 'concept_weighted_ce_loss':
                    loss = self.loss_fn(c_pred, c, 
                                        concept_weights=self.config.cbm.concept_weights, 
                                        device=self.config.device)
                if self.config.loss == 'concept_unweighted_ce_loss':
                    loss = self.loss_fn(c_pred, c, 
                                        device=self.config.device)
                if self.config.loss == 'concept_mse_loss':
                    loss = self.loss_fn(c_pred, c, device=self.config.device)
                if self.config.loss == 'ce_and_concept_mse_loss':
                    loss = self.loss_fn(y_pred, y, c_pred, c, device=self.config.device)
                if self.config.loss == 'cross_entropy':
                    loss = self.loss_fn(y_pred, y)

            cancer_pred.append(y_pred.cpu().detach().numpy())
            cancer_true.append(y.cpu().unsqueeze(-1).detach().numpy())
            concepts_pred.append(c_pred.cpu().detach().numpy())
            concepts_true.append(c.cpu().detach().numpy())

            loss_per_step = np.concatenate([loss_per_step, loss.detach().cpu().item()], axis=None)
            
            # log metrics
            step_metrics = {}
            
            main_lr = self.config.optimizer.main_lr
            step_metrics["main_lr"] = main_lr

            wandb.log(step_metrics)

        cancer_pred = np.vstack(cancer_pred)
        cancer_true = np.vstack(cancer_true)
        cancer_true = np.vstack(cancer_true) # this is not a bug; needs to be stacked twice
        cancer_prob = cancer_pred.copy()
        cancer_pred = np.argmax(cancer_pred, axis=1) 
        
        concepts_pred = np.vstack(concepts_pred)
        concepts_pred = np.where(concepts_pred >= 0.5, 1, 0)
        concepts_true = np.vstack(concepts_true)

        # compute and log metrics
        concept_metrics = compute_conceptwise_metrics(concepts_true, 
                                                      concepts_pred, 
                                                      selected_concepts=self.config.cbm.concepts,
                                                      dataset=self.config.data.dataset,
                                                      desc=desc)
        perf_metrics = compute_classification_metrics(cancer_true, 
                                                      cancer_prob, 
                                                      multi_class=self.config.cbm.num_classes > 2,
                                                      tune_threshold=self.config.metrics.tune_threshold)
        # compute and log metrics
        epoch_metrics = {
            "epoch": self.epoch,
            f"{desc}/loss": np.mean(loss_per_step),
        }

        for (metric_name, metric_value) in concept_metrics.items():
            epoch_metrics[f'{desc}/{metric_name}'] = metric_value
        for (metric_name, metric_value) in perf_metrics.items():
            epoch_metrics[f'{desc}/{metric_name}'] = metric_value

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
            },
            self.exp_state_path,
        )

    def save_model_weights(self, score, is_best_score=False):
        if self.config.checkpoint_dir is None:
            return

        if not is_best_score:
            fname = f"model_epoch.ckpt"
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


