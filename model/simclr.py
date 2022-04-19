#!/usr/bin/env python

# The MIT License (MIT)
# Copyright (c) 2022 NoName
# Paper: "Rethinking of asking help from My Friends: Relation based Nearest-Neighbor Contrastive Learning of Visual Representations", NoName
# GitHub: https://github.com/HuangChiEn/RNN_CLVR
#
# Implementation of the paper:
# "A Simple Framework for Contrastive Learning of Visual Representations", Chen et al. (2020)
# Paper: https://arxiv.org/abs/2002.05709
# Code (adapted from):
# https://github.com/pietz/simclr
# https://github.com/google-research/simclr

import math
import time
import collections

import pytorch_lightning as pl
from torch.optim import SGD, Adam
import torch.nn.functional as F
from torch import nn
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
from util_tool.utils import AverageMeter
               
               
class SimCLR(pl.LightningModule):

    def __init__(self, feature_extractor, temperature):
        self.backbone = feature_extractor
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )
        self.classifier = nn.Linear(self.features_dim, num_classes)
        self.temperature = temperature

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, X: torch.tensor) -> Dict[str, Any]:
        """Performs the forward pass of the backbone and the projector.
        Args:
            X (torch.Tensor): a batch of images in the tensor format.
        Returns:
            Dict[str, Any]:
                a dict containing the outputs of the parent
                and the projected features.
        """
        if not self.no_channel_last:
            X = X.to(memory_format=torch.channels_last)

        feats = self.backbone(X)
        z = self.projector(feats)
        logits = self.classifier(feats.detach())

        return {'z':z, 'logits':logits}
        
    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for SimCLR reusing BaseMethod training step.
        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.
        Returns:
            torch.Tensor: total loss composed of SimCLR loss and classification loss.
        """
        indexes, X, target = batch
        # for support num_n_crop function
        X = [X] if isinstance(X, torch.Tensor) else X
        # but we does not support multicrop now ~
        tmp_outs = list()
        for x in X:         # perform forward function to each view 
            out = self(x)   # default V*B=2B, B:batch, V:view
            tmp_outs.append(out)
        # merge all outputs according to the same key
        outs = {k: [out[k] for out in tmp_outs] for k in tmp_outs[0].keys()}
        z = torch.cat(out["z"])

        # ------- contrastive loss -------
        #n_augs = self.num_large_crops + self.num_small_crops
        #indexes = indexes.repeat(n_augs)

        z = gather(z)
        indexes = gather(indexes)

        nce_loss = simclr_loss_func(
            z,
            indexes=indexes,
            temperature=self.temperature,
        )

        self.log("train_nce_loss", nce_loss, on_epoch=True, sync_dist=True)

        return nce_loss + class_loss
