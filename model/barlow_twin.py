#!/usr/bin/env python

# The MIT License (MIT)
# Copyright (c) 2022 NoName
# Paper: "Rethinking of asking help from My Friends: Relation based Nearest-Neighbor Contrastive Learning of Visual Representations", NoName
# GitHub: https://github.com/HuangChiEn/Lightweight_SSL
#
# Implementation of the paper:
# "A Simple Framework for Contrastive Learning of Visual Representations", Chen et al. (2020)
# Paper: https://arxiv.org/abs/2002.05709
# Code (adapted from):
# https://github.com/vturrisi/solo-learn/
# https://theaisummer.com/simclr/

import math
import time
import collections

import pytorch_lightning as pl
from torch.optim import Adam
import torch.nn.functional as F
from torch import nn, Tensor
import torchvision.datasets as dset
import torchvision.transforms as transforms

from model.losses.barlow_twin_loss import BarlowTwinLoss
from util_tool.utils import dist_gather, accuracy_at_k
               
               
class Barlow_Twin(pl.LightningModule):

    def __init__(self, backbone, proj_hidden_dim, proj_output_dim, lamb, scale_loss, num_of_cls):
        super().__init__()
        self.save_hyperparameters()
        
        self.backbone = backbone
        self.projector = nn.Sequential(collections.OrderedDict([
            ("linear1", nn.Linear(self.backbone.inplanes, proj_hidden_dim, bias=False)),  
            ("bn1", nn.BatchNorm1d(proj_hidden_dim)),
            ("relu1",   nn.ReLU()),
            ("linear2", nn.Linear(proj_hidden_dim, proj_hidden_dim, bias=False)), 
            ("bn2", nn.BatchNorm1d(proj_hidden_dim)),
            ("relu2",   nn.ReLU()),
            ("linear3", nn.Linear(proj_hidden_dim, proj_output_dim)),  
            ("bn3", nn.BatchNorm1d(proj_output_dim))
        ]))
        
        self.classifier = nn.Linear(self.backbone.inplanes, num_of_cls)
        self.loss_fn = BarlowTwinLoss(lamb, scale_loss)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-4)

    def forward(self, X, targets):
        """Performs the forward pass of the backbone and the projector.
        Args:
            X (torch.Tensor): a batch of images in the tensor format.
        Returns:
            Dict[str, Any]:
                a dict containing the outputs of the parent
                and the projected features.
        """
        feats = self.backbone(X)
        z = self.projector(feats)
        
        # handle the linear protocol during the training
        logits = self.classifier(feats.detach())
        loss = F.cross_entropy(logits, targets, ignore_index=-1)
        top_k_max = min(5, logits.size(1))
        acc1, acc5 = accuracy_at_k(logits, targets, top_k=(1, top_k_max))
        
        return {"z":z, "loss": loss, "acc1": acc1, "acc5": acc5}

        
    def training_step(self, batch, batch_idx):
        """Training step for SimSiam reusing BaseMethod training step."""
        X, targets = batch
        # for support num_n_crop function
        X = [X] if isinstance(X, Tensor) else X
        # but we does not support multicrop now ~
        tmp_outs = list()
        for x in X:         # perform forward function to each view 
            out = self(x, targets)   # default V*B=2B, B:batch, V:view
            tmp_outs.append(out)
        
        # merge all outputs according to the same key
        outs = {k: [out[k] for out in tmp_outs] for k in tmp_outs[0].keys()}
        z1, z2 = outs["z"]
        
        # ------- barlow twins loss -------
        barlow_loss = self.loss_fn(z1, z2)
        self.log("train_barlow_loss", barlow_loss, on_epoch=True, sync_dist=True)

        n_viw = len(outs["loss"])
        clf_loss = outs["loss"] = sum(outs["loss"]) / n_viw
        outs["acc1"] = sum(outs["acc1"]) / n_viw
        outs["acc5"] = sum(outs["acc5"]) / n_viw
        metrics = {  # record the linear protocol results
            "lin_loss": outs["loss"],
            "lin_acc1": outs["acc1"],
            "lin_acc5": outs["acc5"]
        }
        self.log_dict(metrics, on_step=True, on_epoch=False, sync_dist=True, prog_bar=True)

        return barlow_loss + outs["loss"]
