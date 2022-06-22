#!/usr/bin/env python
# Implementation of the paper:
# "A Simple Framework for Contrastive Learning of Visual Representations", Chen et al. (2020)
# Paper: https://arxiv.org/abs/2002.05709
# Code (adapted from):
# https://github.com/vturrisi/solo-learn/


import math
import time
import collections

import torch
import pytorch_lightning as pl
from torch.optim import Adam
import torch.nn.functional as F
from torch import nn, Tensor
import torchvision.datasets as dset
import torchvision.transforms as transforms

from model.losses.nnclr_loss import NearestNeighborLoss
from util_tool.utils import dist_gather, accuracy_at_k
               
               
class NN_CLR(pl.LightningModule):
    '''arxiv source:https://arxiv.org/pdf/2104.14548.pdf'''
    #queue: torch.Tensor

    def __init__(self, backbone, queue_size, proj_hidden_dim, proj_output_dim, pred_hidden_dim, temperature, num_of_cls):
        super().__init__()
        self.queue_size = queue_size
        self.backbone = backbone
        # 3 FC layer [2048, 2048, d], hidden layers follow BN & Relu, the last one doesn't have Relu
        self.projector = nn.Sequential(collections.OrderedDict([
            ("linear1", nn.Linear(self.backbone.inplanes, proj_hidden_dim)),  
            ("bn1", nn.BatchNorm1d(proj_hidden_dim)),
            ("relu1",   nn.ReLU()),
            ("linear2", nn.Linear(proj_hidden_dim, proj_hidden_dim)), 
            ("bn2", nn.BatchNorm1d(proj_hidden_dim)),
            ("relu2",   nn.ReLU()),
            ("linear3", nn.Linear(proj_hidden_dim, proj_output_dim)),  
            ("bn3", nn.BatchNorm1d(proj_output_dim))
        ]))
        # 2 FC layer [4096, d], hidden layer follow BN & Relu, the last one doesn't
        self.predictor = nn.Sequential(collections.OrderedDict([
            ("linear1", nn.Linear(proj_output_dim, pred_hidden_dim)),  
            ("bn1", nn.BatchNorm1d(pred_hidden_dim)),
            ("relu1",   nn.ReLU()),
            ("linear2", nn.Linear(pred_hidden_dim, proj_output_dim))
        ]))
        # Nearest-Neighbor support set (FIFO queue)
        self.register_buffer("queue", torch.randn(self.queue_size, proj_output_dim))
        self.register_buffer("queue_y", -torch.ones(self.queue_size, dtype=torch.long))
        self.queue = F.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.classifier = nn.Linear(self.backbone.inplanes, num_of_cls)
        self.loss_fn = NearestNeighborLoss(temperature=temperature)
        
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

    @torch.no_grad()
    def dequeue_and_enqueue(self, z, y):
        """Adds new samples and removes old samples from the queue in a fifo manner. Also stores
        the labels of the samples.
        Args:
            z (torch.Tensor): batch of projected features.
            y (torch.Tensor): labels of the samples in the batch.
        """
        z = dist_gather(z)
        y = dist_gather(y)

        batch_size = z.shape[0]

        ptr = int(self.queue_ptr)  
        assert self.queue_size % batch_size == 0

        self.queue[ptr : ptr + batch_size, :] = z
        self.queue_y[ptr : ptr + batch_size] = y  
        ptr = (ptr + batch_size) % self.queue_size

        self.queue_ptr[0] = ptr  

    @torch.no_grad()
    def find_nn(self, z):
        """Finds the nearest neighbor of a sample.
        Args:
            z (torch.Tensor): a batch of projected features.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                indices and projected features of the nearest neighbors.
        """
        idx = (z @ self.queue.T).max(dim=1)[1]
        nn = self.queue[idx]
        return idx, nn


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
        z = F.normalize(z, dim=-1)
        
        # handle the linear protocol during the training
        logits = self.classifier(feats.detach())
        loss = F.cross_entropy(logits, targets, ignore_index=-1)
        top_k_max = min(5, logits.size(1))
        acc1, acc5 = accuracy_at_k(logits, targets, top_k=(1, top_k_max))
        
        return {"z":z, "p":p, "loss": loss, "acc1": acc1, "acc5": acc5}

        
    def training_step(self, batch, batch_idx):
        """Training step for NNCLR reusing BaseMethod training step."""
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

        # find nn
        idx1, nn1 = self.find_nn(z1)
        _, nn2 = self.find_nn(z2)
        nnclr_loss = (
            self.loss_fn(nn1, p2) / 2
            + self.loss_fn(nn2, p1) / 2
        )
        # compute nn accuracy
        b = targets.size(0)
        nn_acc = (targets == self.queue_y[idx1]).sum() / b

        # dequeue and enqueue
        self.dequeue_and_enqueue(z1, targets)

        n_viw = len(outs["loss"])
        clf_loss = outs["loss"] = sum(outs["loss"]) / n_viw
        outs["acc1"] = sum(outs["acc1"]) / n_viw
        outs["acc5"] = sum(outs["acc5"]) / n_viw
        metrics = {  # record the linear protocol results
            #"lin_loss": outs["loss"],
            "lin_acc1": outs["acc1"],
            "lin_acc5": outs["acc5"],
            #"nnclr_loss" : nnclr_loss,
            "nn_acc" : nn_acc
        }
        self.log_dict(metrics, on_step=True, on_epoch=False, sync_dist=True, prog_bar=True)

        return nnclr_loss + outs["loss"]
    
    ## Progressbar adjustment of output console
    def on_epoch_start(self):
        print('\n')

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        tqdm_dict.pop("loss", None)
        return tqdm_dict