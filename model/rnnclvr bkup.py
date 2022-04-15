#!/usr/bin/env python

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


class FocalLoss(torch.nn.Module):
    """Sigmoid focal cross entropy loss.
    Focal loss down-weights well classified examples and focusses on the hard
    examples. See https://arxiv.org/pdf/1708.02002.pdf for the loss definition.
    """

    def __init__(self, gamma=2.0, alpha=0.25):
        """Constructor.
        Args:
            gamma: exponent of the modulating factor (1 - p_t)^gamma.
            alpha: optional alpha weighting factor to balance positives vs negatives,
                with alpha in [0, 1] for class 1 and 1-alpha for class 0. 
                In practice alpha may be set by inverse class frequency,
                so that for a low number of positives, its weight is high.
        """
        super(FocalLoss, self).__init__()
        self._alpha = alpha
        self._gamma = gamma
        self.BCEWithLogits = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, prediction_tensor, target_tensor):
        """Compute loss function.
        Args:
            prediction_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing the predicted logits for each class
            target_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing one-hot encoded classification targets.
        Returns:
            loss: a float tensor of shape [batch_size, num_anchors, num_classes]
            representing the value of the loss function.
        """
        per_entry_cross_ent = self.BCEWithLogits(prediction_tensor, target_tensor)
        prediction_probabilities = torch.sigmoid(prediction_tensor)
        p_t = ((target_tensor * prediction_probabilities) + #positives probs
                ((1 - target_tensor) * (1 - prediction_probabilities))) #negatives probs
        modulating_factor = 1.0
        if self._gamma:
            modulating_factor = torch.pow(1.0 - p_t, self._gamma) #the lowest the probability the highest the weight
        alpha_weight_factor = 1.0
        if self._alpha is not None:
            alpha_weight_factor = (target_tensor * self._alpha + (1 - target_tensor) * (1 - self._alpha))
        focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor * per_entry_cross_ent)
        return torch.mean(focal_cross_entropy_loss)
                    

class RNN_CLVR(pl.LightningModule):
    
    def __init__(self, feature_extractor, aggregation="cat", queue_size=65536, gpu_str=''):
        super(RNN_CLVR, self).__init__()

        self.op_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # encoder
        # note that, wo predictor MLP decrease 0.4 top-1, 0.1 top-5 acc
        self.net = feature_extractor
        ## I. relation reasonning
        self.aggregation=aggregation
        if(self.aggregation=="cat"): resizer=2
        elif(self.aggregation=="sum"): resizer=1
        elif(self.aggregation=="mean"): resizer=1
        elif(self.aggregation=="max"): resizer=1
        else: RuntimeError("[ERROR] aggregation type " + str(self.aggregation) +  " not supported, must be: cat, sum, mean.")
        
        # projector 
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

        # predictor
        self.predictor = nn.Sequential(collections.OrderedDict([
          ("linear1",  nn.Linear(256, 512)),
          ("bn1",      nn.BatchNorm1d(512)),
          ("relu",     nn.LeakyReLU()),
          ("linear2",  nn.Linear(512, 256)),
        ]))

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
        
        id_lst = [0]#id_lst = [int(n.strip()) for n in gpu_str.split(',')]
        if torch.cuda.is_available() and len(id_lst) <= torch.cuda.device_count():
            print('[INFO]: Active data parallel\n')
            print(f'[INFO]: Device ids : {id_lst}\n')
            self.net = torch.nn.DataParallel(self.net, device_ids=id_lst).to(self.op_device)
            self.projector = torch.nn.DataParallel(self.projector, device_ids=id_lst).to(self.op_device)
            self.predictor = torch.nn.DataParallel(self.predictor, device_ids=id_lst).to(self.op_device)
            self.relation_module = torch.nn.DataParallel(self.relation_module, device_ids=id_lst).to(self.op_device)

        else:
            print(f'[INFO]: single device ({self.op_device})  mode\n')
            self.net = self.net
            self.relation_module = self.relation_module

    
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
        batch_embs = batch_embs.to(self.op_device) ; embed_queue = self.embed_queue.to(self.op_device)
        cos_sim_mtr = batch_embs @ embed_queue.T
        
        idx = (cos_sim_mtr).max(dim=1)[1]
        nn = embed_queue[idx]
        return nn
    
        
    def relation_aggregate(self, features, tot_augmentations, type="cat"):
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
        relation_pairs_list = list()
        target_list = list()
        embed_lst = list()
        
        # batch_size
        siz = int(features.shape[0] / tot_augmentations) 
        tot_positive = 0.0
        tot_negative = 0.0
        shifts_counter = 1 

        #features = features / features.norm(dim=1)[:, None]
        # compute nn accuracy
        #b = targets.size(0)
        #nn_acc = (targets == self.queue_y[idx1]).sum() / b

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
                if(type=="cat"): 
                    positive_pair_1 = torch.cat([nn_fea_1, p_2], 1)
                    positive_pair_2 = torch.cat([nn_fea_2, p_1], 1)
                    
                    negative_pair_1 = torch.cat([nn_fea_1, 
                                               torch.roll(p_2, shifts=shifts_counter, dims=0)], 1)
                    negative_pair_2 = torch.cat([nn_fea_2, 
                                               torch.roll(p_1, shifts=shifts_counter, dims=0)], 1)
                    
                #if(type=="cat"): 
                #    positive_pair = torch.cat([nn_fea_1, features[idx_2:idx_2+siz]], 1)
                #    negative_pair = torch.cat([nn_fea_1, 
                #                               torch.roll(features[idx_2:idx_2+siz], shifts=shifts_counter, dims=0)], 1)
                elif(type=="sum"): 
                    positive_pair = nn_fea_1 + features[idx_2:idx_2+siz]
                    negative_pair = nn_fea_1 + torch.roll(features[idx_2:idx_2+siz], shifts=shifts_counter, dims=0)
                elif(type=="mean"): 
                    positive_pair = (nn_fea_1 + features[idx_2:idx_2+siz]) / 2.0
                    negative_pair = (nn_fea_1 + torch.roll(features[idx_2:idx_2+siz], shifts=shifts_counter, dims=0)) / 2.0
                elif(type=="max"):
                    positive_pair, _ = torch.max(torch.stack([nn_fea_1, features[idx_2:idx_2+siz]], 2), 2)
                    negative_pair, _ = torch.max(torch.stack([nn_fea_1, 
                                                              torch.roll(features[idx_2:idx_2+siz], shifts=shifts_counter, dims=0)], 2), 2)
                
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

    def setup_train(self):
        # setup train
        self.net.train()
        self.relation_module.train()
        

    def get_model_dict(self):
        feature_extractor_state_dict = self.net.module.state_dict()
        relation_state_dict = self.relation_module.module.state_dict()
        return {"backbone": feature_extractor_state_dict, "relation": relation_state_dict}

    def load_model(self, checkpoint):
        self.net.feature_extractor.load_state_dict(checkpoint["backbone"])
        self.relation_module.load_state_dict(checkpoint["relation"])

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


## Trainer interface
class Model(object):
    def __init__(self, feature_extractor, aggregation="cat", queue_size=16384, gpu_str=""):
        self.op_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_inst = RNN_CLVR(feature_extractor, aggregation, queue_size, gpu_str)
        # wrap into data_parellel model
        self.__setup_opt_loss()

    def __setup_opt_loss(self):
        self.optimizer = Adam([{"params": self.model_inst.parameters(), "lr": 0.001}])
        self.fl = FocalLoss(gamma=2.0, alpha=0.5)   # Using reccommended value for gamma: 2.0
        #self.fl = nn.BCEWithLogitsLoss() # Standard BCE loss can also be used
        
    def train(self, epoch, train_loader):
        start_time = time.time()
        self.model_inst.setup_train()
        
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