# The MIT License (MIT)
# Copyright (c) 2022 NoName
# Paper: "Rethinking of asking help from My Friends: Relation based Nearest-Neighbor Contrastive Learning of Visual Representations", NoName
# GitHub: https://github.com/HuangChiEn/RNN_CLVR
#
# Implementation of our Relational Reasoning method as described in the paper.
# This code use a Focal Loss but also a standard BCE loss can be used.
# An essential version of this code has also been provided in the repository.

import time
import math
import collections
import numpy as np
import random

import torch
from torch import nn
from torch.optim import SGD, Adam
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
from util_tool.utils import dist_gather, accuracy_at_k
from torch.nn.functional import softmax
from model.losses.focal_loss import FocalLoss
import pytorch_lightning as pl

class RNN_CLVR(pl.LightningModule):
    
    def __init__(self, backbone, aggregation="cat", queue_size=65536, shift_range=2,
                        proj_hidden_dim=2048, proj_output_dim=256, pred_hidden_dim=4096, num_of_cls=100):
        super().__init__()
        # for record the hyparam while enable the checkpoint call back
        self.save_hyperparameters() 
        agg_dict = {'cat' : (2, lambda x, y, dims : torch.cat([x, y], dims)), 'sum' : (1, lambda x, y : (x+y)), 
                         'mean' : (1, lambda x, y : (x+y) / float(num_crop)), 'max' : (1, torch.max)}
        support_type = agg_dict.keys()
        if not aggregation in support_type:
            ValueError(f"[ERROR] aggregation type {aggregation} not supported, must be: {support_type}")

        resizer = agg_dict[aggregation][0]
        self.agg_op = agg_dict[aggregation][1]
        self.shft_rng = shift_range   # default=2 indicate one pos vs. one neg sample (1+1=2)
        self.backbone = backbone
        self.classifier = nn.Linear(self.backbone.inplanes, num_of_cls)

        ## I. relation reasonning
        # relation metric
        self.relation_module = nn.Sequential(collections.OrderedDict([
          ("linear1",  nn.Linear(proj_output_dim*resizer, proj_output_dim)),
          ("bn1",      nn.BatchNorm1d(proj_output_dim)),
          ("relu1",     nn.LeakyReLU()),
          ("linear2",  nn.Linear(proj_output_dim, 1)),   # output unbounded logits (batch_size x logit)
        ]))
        
        ## II. Support set of neighbor embedding 
        self.queue_size = queue_size
        # note that feature embedding should be [128, 256]
        self.register_buffer("queue_y", -torch.ones(self.queue_size, dtype=torch.long))
        self.register_buffer("embed_queue", torch.randn(self.queue_size, proj_output_dim))
        self.embed_queue = F.normalize(self.embed_queue, dim=1, p=2)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        ## III. other option for ssl-framework structure
        self.projector = nn.Sequential(collections.OrderedDict([
            ("linear1", nn.Linear(self.backbone.inplanes, proj_hidden_dim)),  # avgpool2d output 512 dim vec
            ("bn1",     nn.BatchNorm1d(proj_hidden_dim)),
            ("relu1",   nn.LeakyReLU()),
            ("linear2", nn.Linear(proj_hidden_dim, proj_hidden_dim)),
            ("bn2",     nn.BatchNorm1d(proj_hidden_dim)),
            ("relu2",   nn.LeakyReLU()),
            ("linear3", nn.Linear(proj_hidden_dim, proj_output_dim)),
            ("bn3", nn.BatchNorm1d(proj_output_dim)),
        ]))
        
        self.predictor = nn.Sequential(collections.OrderedDict([
            ("linear1",  nn.Linear(proj_output_dim, pred_hidden_dim)),
            ("bn1",      nn.BatchNorm1d(pred_hidden_dim)),
            ("relu1",     nn.LeakyReLU()),
            ("linear2",  nn.Linear(pred_hidden_dim, proj_output_dim)),
        ]))
        
        self.loss_fn = nn.CrossEntropyLoss() #FocalLoss(gamma=2.0, alpha=0.5) if focal else nn.BCEWithLogitsLoss()
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    @torch.no_grad()
    def dequeue_and_enqueue(self, z, y):
        """Adds new samples and removes old samples from the queue in a FIFO manner. 
        Args:
            z (torch.Tensor): batch of projected features.
        """
        z, y = dist_gather(z), dist_gather(y)
        batch_size = z.shape[0]
        ptr = int(self.queue_ptr)
        assert z.shape[0] == y.shape[0], f"num of lab : {y.shape[0]} not match with num of embeds : {z.shape[0]}"
        assert self.queue_size % batch_size == 0, f"que siz : {self.queue_size} not match batch siz : {batch_size}\n"

        self.embed_queue[ptr : ptr + batch_size, :] = z 
        self.queue_y[ptr : ptr + batch_size] = y
        
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr 
        
    @torch.no_grad()
    def cos_find_nn(self, batch_embs, sample_ratio=1.0):
        """Finds the nearest neighbor of a sample by cos-distance
        Args:
            z (torch.Tensor): a batch of projected features.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                indices and projected features of the nearest neighbors.
        """
        cos_sim_mtr = batch_embs @ self.embed_queue.T
        
        idx = (cos_sim_mtr).max(dim=1)[1]
        nn = self.embed_queue[idx]
        return idx, nn

    def nn_relation_estimation(self, Z, P):
        (z1, z2), (p1, p2) = Z, P     # Z[0] : view_1, Z[1] : view_2
        idx, nn_z1 = self.cos_find_nn(z1)  # (32, 3, 32, 32)
        _, nn_z2 = self.cos_find_nn(z2)  # (32, 3, 32, 32)

        logit_lst1 = list()
        logit_lst2 = list()
        batch_size = self.shft_rng if self.shft_rng else z1.shape[0]
        for idx in range(0, batch_size):  # idx == 0 as positive sample
            p1 = torch.roll(p1, shifts=idx, dims=0)
            p2 = torch.roll(p2, shifts=idx, dims=0)
            rel_pair1 = self.agg_op(nn_z1, p2, 1)
            rel_pair2 = self.agg_op(nn_z2, p1, 1) # symmetric terms
            logit_lst1.append( self.relation_module(rel_pair1) )
            logit_lst2.append( self.relation_module(rel_pair2) )

        return logit_lst1, logit_lst2, idx

    def logit2prob(self, logit_lst):
        logit_mtr = torch.cat(logit_lst, 1)
        return softmax(logit_mtr, dim=1)  # out_mtr[0, :] is prb_vec with first sample [pos_11, neg_12, neg_13, ..., neg_1n]

    def forward(self, X, targets):
        feats = self.backbone(X)
        z = self.projector(feats)
        p = self.predictor(z)
        z = F.normalize(z, dim=-1)
        outs = {"z": z, "p": p}
        
        # handle the linear protocol during the training
        logits = self.classifier(feats.detach())
        loss = F.cross_entropy(logits, targets, ignore_index=-1)
        top_k_max = min(5, logits.size(1))
        acc1, acc5 = accuracy_at_k(logits, targets, top_k=(1, top_k_max))
        outs.update({"loss": loss, "acc1": acc1, "acc5": acc5})
        return outs


    def training_step(self, batch, batch_idx):
        X, targets = batch
        # for support num_n_crop function
        X = [X] if isinstance(X, torch.Tensor) else X
        # but we does not support multicrop now ~
        tmp_outs = list()
        for x in X:  # perform forward function to each view (defualt : 2 view)
            out = self(x, targets)  # diff view have same targets
            tmp_outs.append(out)
        # merge all outputs according to the same key
        outs = {k: [out[k] for out in tmp_outs] for k in tmp_outs[0].keys()}
        n_viw = len(outs["loss"])
        clf_loss = outs["loss"] = sum(outs["loss"]) / n_viw
        outs["acc1"] = sum(outs["acc1"]) / n_viw
        outs["acc5"] = sum(outs["acc5"]) / n_viw
        metrics = {  # record the linear protocol results
            "lin_loss": outs["loss"],
            "lin_acc1": outs["acc1"],
            "lin_acc5": outs["acc5"]
        }
        
        logit_lst1, logit_lst2, idx = \
                        self.nn_relation_estimation(outs['z'], outs['p'])
        
        prb_v1, prb_v2 = self.logit2prob(logit_lst1), self.logit2prob(logit_lst2)
        gt_prb = torch.zeros([prb_v1.shape[0]], dtype=torch.long).to('cuda') # to device is needed, lighting bug.. 
        
        # compute nn accuracy
        b = targets.shape[0]
        nn_acc = (targets == self.queue_y[idx]).sum() / b
        
        # BYOL style symmetric terms 
        rnnclvr_loss = self.loss_fn(prb_v1, gt_prb) / 2 \
                        + self.loss_fn(prb_v2, gt_prb) / 2
        metrics.update({"tra_loss": rnnclvr_loss, "nn_acc": nn_acc})
        self.log_dict(metrics,  on_step=True, on_epoch=False, sync_dist=True, prog_bar=True)
        
        # does not support evaluation of nn_support set acc
        self.dequeue_and_enqueue(outs['z'][0], targets)
        
        return  rnnclvr_loss + clf_loss
        
    ## Progressbar adjustment of output console
    def on_epoch_start(self):
        print('\n')

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        tqdm_dict.pop("loss", None)
        return tqdm_dict