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
from torch import optim
import torch.nn.functional as F
from torch import nn, Tensor
import torchvision.datasets as dset
import torchvision.transforms as transforms

from model.losses.simsiam_loss import SimSiamLoss
#from model.solver.lr_scheduler import LARS, 
from util_tool.utils import dist_gather, accuracy_at_k
               
               
class Sim_Siam(pl.LightningModule):

    def __init__(self, backbone, proj_hidden_dim, proj_output_dim, pred_hidden_dim, num_of_cls):
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
            ("linear3", nn.Linear(proj_hidden_dim, proj_output_dim)),   # , affines=False
            ("bn3", nn.BatchNorm1d(proj_output_dim))
        ]))
        self.projector.linear3.bias.requires_grad = False  # hack: not use bias as it is followed by BN
        
        self.predictor = nn.Sequential(collections.OrderedDict([
            ("linear1", nn.Linear(proj_output_dim, pred_hidden_dim)),  
            ("bn1", nn.BatchNorm1d(pred_hidden_dim)),
            ("relu1",   nn.ReLU()),
            ("linear2", nn.Linear(pred_hidden_dim, proj_output_dim))
        ]))
        
        self.classifier = nn.Linear(self.backbone.inplanes, num_of_cls)
        self.loss_fn = SimSiamLoss()

    def configure_optimizers(self):
        # linear scaling rule : 0.05*Batchsize / 256 = 0.05 (default)
        optimizer = optim.SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

        return [optimizer], [scheduler]

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
        p = self.predictor(z)
        
        # handle the linear protocol during the training
        logits = self.classifier(feats.detach())
        loss = F.cross_entropy(logits, targets, ignore_index=-1)
        top_k_max = min(5, logits.size(1))
        acc1, acc5 = accuracy_at_k(logits, targets, top_k=(1, top_k_max))
        
        return {"z":z, "p":p, "loss": loss, "acc1": acc1, "acc5": acc5}

        
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
        p1, p2 = outs["p"]
        
        # ------- contrastive loss -------
        neg_cos_sim = self.loss_fn(p1, z2) / 2 + self.loss_fn(p2, z1) / 2

        # calculate std of features
        z1_std = F.normalize(z1, dim=-1).std(dim=0).mean()
        z2_std = F.normalize(z2, dim=-1).std(dim=0).mean()
        z_std = (z1_std + z2_std) / 2

        metrics = {
            "train_neg_cos_sim": neg_cos_sim,
            "train_z_std": z_std,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

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

        return neg_cos_sim + outs["loss"]

    ## Progressbar adjustment of output console
    def on_epoch_start(self):
        print('\n')

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        tqdm_dict.pop("loss", None)
        return tqdm_dict