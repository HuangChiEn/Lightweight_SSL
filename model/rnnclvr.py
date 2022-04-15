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
from util_tool.utils import AverageMeter  #, gather
from model.losses.focal_loss import FocalLoss
                    

class RNN_CLVR(pl.LightningModule):
    
    def __init__(self, feature_extractor, aggregation="cat", queue_size=65536,
        enable_proj=False, enbale_pred=False, focal=True):
        super().__init__()

        self.agg_dict = {'cat' : (2, torch.cat), 'sum' : (1, lambda x, y : (x+y)), 
                         'mean' : (1, lambda x, y : (x+y) / 2.0), 'max' : (1, torch.max)}
        support_type = self.agg_dict.keys()
        if not aggregation in support_type:
            ValueError(f"[ERROR] aggregation type {aggregation} not supported, must be: {support_type}")

        resizer = self.agg_dict[aggregation][0]
        self.aggregation = aggregation
        self.net = feature_extractor

        ## I. relation reasonning
        # relation metric
        self.relation_module = nn.Sequential(collections.OrderedDict([
          ("linear1",  nn.Linear(256*resizer, 256)),
          ("bn1",      nn.BatchNorm1d(256)),
          ("relu",     nn.LeakyReLU()),
          ("linear2",  nn.Linear(256, 1)),
        ]))

        ## II. Support set of neighbor embedding 
        self.queue_size = queue_size
        # note that feature embedding should be [128, 256] (same performance!)
        self.register_buffer("embed_queue", torch.randn(self.queue_size, 256)) #feature_extractor.feature_size))
        self.embed_queue = F.normalize(self.embed_queue, dim=1, p=2)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        ## III. other option for ssl-framework structure
        self.enable_proj = enable_proj # python 3.10 waluform operator can be used
        if enable_proj:
            self.projector = nn.Sequential(collections.OrderedDict([
                ("linear1", nn.Linear(self.net.fc, 256)),
                ("bn1",     nn.BatchNorm1d(256)),
                ("relu1",   nn.ReLU()),
                ("linear2", nn.Linear(256, 256)),
                ("bn2",     nn.BatchNorm1d(256)),
                ("relu2",   nn.ReLU()),
                ("linear2", nn.Linear(256, 256)),
                ("bn3", nn.BatchNorm1d(256)),
            ]))
        self.enable_pred = enable_pred
        if enbale_pred:
            self.predictor = nn.Sequential(collections.OrderedDict([
            ("linear1",  nn.Linear(256, 512)),
            ("bn1",      nn.BatchNorm1d(512)),
            ("relu",     nn.LeakyReLU()),
            ("linear2",  nn.Linear(512, 256)),
            ]))

        self.loss_fn = FocalLoss(gamma=2.0, alpha=0.5) if focal else nn.BCEWithLogitsLoss()
    

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)
        

    @torch.no_grad()
    def dequeue_and_enqueue(self, z: torch.Tensor):
        """Adds new samples and removes old samples from the queue in a FIFO manner. 
        Args:
            z (torch.Tensor): batch of projected features.
        """
        z = gather( torch.cat(z) )
        batch_size = z.shape[0]
        
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0, f"que siz : {self.queue_size} not match batch siz : {batch_size}\n"

        self.embed_queue[ptr : ptr + batch_size, :] = z 
        
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr 
        

    @torch.no_grad()
    def cos_find_nn(self, batch_embs: torch.Tensor, sample_ratio=1.0):
        """Finds the nearest neighbor of a sample by cos-distance
        Args:
            z (torch.Tensor): a batch of projected features.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                indices and projected features of the nearest neighbors.
        """
        cos_sim_mtr = batch_embs @ self.embed_queue.T
        
        idx = (cos_sim_mtr).max(dim=1)[1]
        nn = embed_queue[idx]
        return nn
    
        
    def relation_aggregate(self, features, tot_augmentations):
        """Aggregation function.
        Args:
          features: The features returned by the backbone, it is a tensor
            of shape [batch_size*K, feature_size].
            num_classes] representing the predicted logits for each class
          tot_augmentations: The total number of augmentations, corresponds
            to the parameter K in the paper.
        Returns:
          relation_pairs: a tensor with the aggregated pairs that can be
            given as input to the relation head. 
          target: the values (zeros and ones) for each pair, that
            represents the target used to train the relation head.
          tot_positive: Counter for the total number of positives.
          tot_negative: Counter for the total number of negatives.
        """
        agg_op = self.agg_dict[self.aggregation][1]
        relation_pairs_list = list()
        target_list = list()
        embed_lst = list()
        
        # batch_size
        siz = int(features.shape[0] / tot_augmentations) 
        tot_positive = 0.0
        tot_negative = 0.0
        shifts_counter = 1 

        for idx_1 in range(0, siz*tot_augmentations, siz):
            # Finds the neighbor embeddings by the 'learnable' relation metric
            #fea_1 = features[idx_1:idx_1+siz]
            fea_1 = self.projector(features[idx_1:idx_1+siz])
            p_1 = self.predictor(fea_1)
            fea_1 = F.normalize(fea_1, dim=-1, p=2)
            
            nn_fea_1 = self.cos_find_nn(fea_1)
            embed_lst.append( fea_1.clone().detach() )
            for idx_2 in range(idx_1+siz, siz*tot_augmentations, siz):
                #fea_2 = features[idx_2:idx_2+siz]
                fea_2 = self.projector(features[idx_2:idx_2+siz])
                p_2 = self.predictor(fea_2)
                fea_2 = F.normalize(fea_2, dim=-1, p=2)
                nn_fea_2 = self.cos_find_nn(fea_2)
                embed_lst.append( fea_2.clone().detach() )
                
                # perform aggregation operation
                positive_pair_1 = agg_op([nn_fea_1, p_2], 1)
                positive_pair_2 = agg_op([nn_fea_2, p_1], 1)
                
                negative_pair_1 = agg_op([nn_fea_1, 
                                            torch.roll(p_2, shifts=shifts_counter, dims=0)], 1)
                negative_pair_2 = agg_op([nn_fea_2, 
                                            torch.roll(p_1, shifts=shifts_counter, dims=0)], 1)
                

                relation_pairs_list.append(positive_pair_1)
                relation_pairs_list.append(positive_pair_2)
                relation_pairs_list.append(negative_pair_1)
                relation_pairs_list.append(negative_pair_2)

                target_list.append(torch.ones(siz, dtype=torch.float32))
                target_list.append(torch.ones(siz, dtype=torch.float32))
                target_list.append(torch.zeros(siz, dtype=torch.float32))
                target_list.append(torch.zeros(siz, dtype=torch.float32))

                tot_positive += siz 
                tot_negative += siz
                shifts_counter+=1
                if(shifts_counter>=siz): shifts_counter=1 # reset to avoid neutralizing the roll

        relation_pairs = torch.cat(relation_pairs_list, 0)
        target = torch.cat(target_list, 0)
        return relation_pairs, target, tot_positive, tot_negative, embed_lst


    # forward func for network backbone (Enc, Proj, Pred)
    def forward(self, x):
        fea = self.net(x)
        if self.enable_proj:
            out = self.projector(fea)
        if self.enable_pred:
            out = self.projector(out)

        return out


    def training_step(self, batch, batch_idx):
        metric_duct = {}

        x, _ = batch

        out = self(x)
        if self.enable_pred:
            ...

        metric_duct['loss'] = loss_val = self.loss_fn(y_hat, y)


        # already perform mean at end of epoch
        self.log_dict(metric_duct, {'loss':loss_val, 'acc':acc_val, 'acc+':acc_pos, 'acc-':acc_neg},
                    on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss


    def forward(self, train_x, num_view):
        # norm embeddings & aggregation embeddings based on the relation metric
        features = self.net(train_x)
        relation_pairs, train_y, tot_positive, tot_negative, embed_lst = self.relation_aggregate(features, num_view, type=self.aggregation)
        
        train_y = train_y.to(self.op_device)
        tot_pairs = int(relation_pairs.shape[0])
        # forward of the pairs through the relation head
        predictions = self.relation_module(relation_pairs).squeeze()
        predictions.to(self.op_device)

        # update the queue of embedding support set 
        self.dequeue_and_enqueue(embed_lst)
        
        return {"predictions":predictions, "train_y":train_y, 
                "tot_pairs":tot_pairs, "tot_positive":tot_positive, "tot_negative":tot_negative}


    def train_step(self, epoch, train_loader):
        start_time = time.time()
        
        accuracy_pos_list = list()
        accuracy_neg_list = list()
        loss_meter = AverageMeter()
        accuracy_meter = AverageMeter()
        statistics_dict = {}
        # aug_views : (num_view, batch_size, C, H, W), num_view : 2 in normal case
        for i, (src_im, aug_views, _) in enumerate(train_loader):
            batch_size = src_im.shape[0]
            num_view = len(aug_views)

            # flatten aug_views into (num_view*batch_size, C, H, W)
            train_x = torch.cat(aug_views, 0).to(self.op_device)
            self.optimizer.zero_grad()
            out_dict = self.model_inst( **dict(train_x=train_x, num_view=num_view) )
            # estimate the focal loss (also standard BCE can be used here)
            loss = self.fl(out_dict['predictions'], out_dict['train_y'])
            loss_meter.update(loss.item(), len(out_dict['train_y']))
            # backward step and weights update
            loss.backward()
            self.optimizer.step()
            
            best_guess = torch.round(torch.sigmoid(out_dict['predictions']))
            correct = best_guess.eq(out_dict['train_y'].view_as(best_guess))
            correct_positive = correct[0:int(len(correct)/2)].cpu().sum()
            correct_negative = correct[int(len(correct)/2):].cpu().sum()
            correct = correct.cpu().sum()
            accuracy = (100.0 * correct / float(len(out_dict['train_y']))) 
            accuracy_meter.update(accuracy.item(), len(out_dict['train_y']))
            accuracy_pos_list.append((100.0 * correct_positive / float(len(out_dict['train_y'])/2)).item())
            accuracy_neg_list.append((100.0 * correct_negative / float(len(out_dict['train_y'])/2)).item())
            
            if(i==0): 
                statistics_dict["batch_size"] = batch_size
                statistics_dict["tot_pairs"] = out_dict['tot_pairs']
                statistics_dict["tot_positive"] = int(out_dict['tot_positive'])
                statistics_dict["tot_negative"] = int(out_dict['tot_negative'])
            
        elapsed_time = time.time() - start_time
        
        # Here we are printing a rich set of information to monitor training.
        # The accuracy over both positive and negative pairs is printed separately.
        # The batch-size and total number of pairs is printed for debugging.
        print("Epoch [" + str(epoch) + "]"
                + "[" + str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))) + "]"
                + " loss: " + str(loss_meter.avg)
                + "; acc: " + str(accuracy_meter.avg) + "%"
                + "; acc+: " + str(round(np.mean(accuracy_pos_list), 2)) + "%"
                + "; acc-: " + str(round(np.mean(accuracy_neg_list), 2)) + "%"
                + "; batch-size: " + str(statistics_dict["batch_size"])
                + "; tot-pairs: " + str(statistics_dict["tot_pairs"]))
        return loss_meter.avg, accuracy_meter.avg

    def save(self, file_path="./checkpoint.dat"):
        save_dict = self.model_inst.get_model_dict()
        save_dict["optimizer"] = self.optimizer.state_dict()
        torch.save(save_dict, file_path)
        
    def load(self, file_path):
        checkpoint = torch.load(file_path)
        if("optimizer" in checkpoint): 
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            print("[INFO][RelationNet] Loaded optimizer state-dict")
        self.model_inst.load_model(checkpoint)