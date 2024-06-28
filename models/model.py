# Base / Native
import csv
from collections import Counter
import copy
import json
import functools
import gc
import logging
import math
import os
import pdb
import pickle
import random
import sys
import time

# Numerical / Array
import numpy as np

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor, bilinear
from torch.autograd import Variable
from torch.nn import init, Parameter
from torch.utils.data import DataLoader
from torch.utils.model_zoo import load_url as load_state_dict_from_url
import torch.optim.lr_scheduler as lr_scheduler


# Env
from .fusion import *
from utils.utils import *
# from .resnets import ResNet18
from .mil import *
from .DeformCrossTransMIL import DeformCrossTransMIL
from .MultiheadAttention import *

# from my_utils.compute_gradients import get_grad_embedding

# pre-trained model of MobileNetV2
import torchvision

################
# Network Utils
################
def define_net(args, path_only = False, omic_only = False):
    net = None
    act = define_act_layer(act_type=args.act_type)
    init_max = True if args.init_type == "max" else False

    if args.mode == "path":
        print("creating path model")
        net = ABMIL(args)
        # net = TransMIL(args)
    elif args.mode == "omic":
        print("creating omic model")
        net = MaxNet(input_dim=args.input_size_omic, omic_dim=args.omic_dim, return_grad = args.return_grad,
                               dropout_rate=args.dropout_rate, label_dim=args.label_dim, init_max=init_max)
    elif args.mode == "pathomic":
        print("creating pathomic model")
        net = PathomicNet(args=args, act=act)
    elif args.mode == "pathomic_original":
        print("creating pathomic_original model")
        net = PathomicNet_Original(args=args, act=act)
    elif args.mode == 'mcat':
        print("creating mcat model")
        net = MCAT_Surv(args=args)
    elif args.mode == 'cmta':
        print("creating cmta model")
        net = CMTA(args=args)
    elif args.mode == "deformpathomic":
        print("creating deformpathomic model")
        net = DeformPathomicNet(args=args)
    else:
        raise NotImplementedError('model [%s] is not implemented' % args.model)
    return init_net(net, args.init_type, args.init_gain)


def define_optimizer(args, model):
    optimizer = None
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, initial_accumulator_value=0.1)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % args.optimizer)
    return optimizer

def define_scheduler(args, optimizer):
    if args.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + args.epoch_count - args.epochs) / float(args.epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'exp':
        scheduler = lr_scheduler.ExponentialLR(optimizer, 0.1, last_epoch=-1)
    elif args.lr_policy == 'step':
       scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_iters, gamma=0.1)
    elif args.lr_policy == 'plateau':
       scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif args.lr_policy == 'cosine':
       scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
    elif args.lr_policy == 'onecycle':
       scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, epochs=args.epochs+args.epochs_decay, steps_per_epoch=200)
    else:
       return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


def define_act_layer(act_type='Tanh'):
    if act_type == 'Tanh':
        act_layer = nn.Tanh()
    elif act_type == 'ReLU':
        act_layer = nn.ReLU()
    elif act_type == 'Sigmoid':
        act_layer = nn.Sigmoid()
    elif act_type == 'LSM':
        act_layer = nn.LogSoftmax(dim=1)
    elif act_type == "none":
        act_layer = None
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act_type)
    return act_layer


def define_bifusion(fusion_type, skip=1, use_bilinear=1, gate1=1, gate2=1, dim1=32, dim2=32, scale_dim1=1, scale_dim2=1, mmhid=32, dropout_rate=0.25):
    fusion = None
    if fusion_type == 'pofusion':
        fusion = BilinearFusion(skip=skip, use_bilinear=use_bilinear, gate1=gate1, gate2=gate2, dim1=dim1, dim2=dim2, scale_dim1=scale_dim1, scale_dim2=scale_dim2, mmhid=mmhid, dropout_rate=dropout_rate)
    else:
        raise NotImplementedError('fusion type [%s] is not found' % fusion_type)
    return fusion

############
# Omic Model
############
class MaxNet(nn.Module):
    def __init__(self, input_dim=59, omic_dim=32, return_grad = 'False', dropout_rate=0.25, label_dim=1, init_max=True):
        super(MaxNet, self).__init__()
        hidden = [64, 48, 32, 32]
        self.return_grad = return_grad

        encoder1 = nn.Sequential(
            nn.Linear(input_dim, hidden[0]),
            # nn.BatchNorm1d(hidden[0]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))
        
        encoder2 = nn.Sequential(
            nn.Linear(hidden[0], hidden[1]),
            # nn.BatchNorm1d(hidden[1]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))
        
        encoder3 = nn.Sequential(
            nn.Linear(hidden[1], hidden[2]),
            # nn.BatchNorm1d(hidden[2]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))

        encoder4 = nn.Sequential(
            nn.Linear(hidden[2], omic_dim),
            # nn.BatchNorm1d(omic_dim),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))
        
        # self.encoder = nn.Sequential(encoder1, encoder2, encoder3, encoder4)
        self.encoder = nn.Sequential(encoder1, encoder2, encoder3, encoder4)
        self.relu = nn.ReLU(inplace=False)
        self.classifier = nn.Sequential(nn.Linear(omic_dim, label_dim))

        if init_max: init_max_weights(self)

        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, **kwargs):
        x = kwargs['x_omic'] #[B, 431]
        features = self.relu(self.encoder(x))
        logits = self.classifier(features)

        return features, logits, None

class MaxNet_noclassifier(nn.Module):
    def __init__(self, input_dim=59, omic_dim=32, return_grad = 'False', dropout_rate=0.25, label_dim=1, init_max=True):
        super(MaxNet_noclassifier, self).__init__()
        hidden = [64, 48, 32, 32]
        self.return_grad = return_grad

        encoder1 = nn.Sequential(
            nn.Linear(input_dim, hidden[0]),
            # nn.BatchNorm1d(hidden[0]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))
        
        encoder2 = nn.Sequential(
            nn.Linear(hidden[0], hidden[1]),
            # nn.BatchNorm1d(hidden[1]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))
        
        encoder3 = nn.Sequential(
            nn.Linear(hidden[1], hidden[2]),
            # nn.BatchNorm1d(hidden[2]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))

        encoder4 = nn.Sequential(
            nn.Linear(hidden[2], omic_dim),
            # nn.BatchNorm1d(omic_dim),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))
        
        # self.encoder = nn.Sequential(encoder1, encoder2, encoder3, encoder4)
        self.encoder = nn.Sequential(encoder1, encoder2, encoder3, encoder4)
        self.relu = nn.ReLU(inplace=False)
        # self.classifier = nn.Sequential(nn.Linear(omic_dim, label_dim))

        if init_max: init_max_weights(self)

        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, **kwargs):
        x = kwargs['x_omic'] #[B, 431]
        features = self.relu(self.encoder(x))
        # logits = self.classifier(features)

        if self.return_grad == "True":
            omic_grads = get_grad_embedding(logits, features).detach().cpu().numpy()
        else:
            omic_grads = None

        return features, features, omic_grads


############
# Path Model
############

##############################################################################
# Path + Omic  ### train the CNN and SNN models simultaneously.
##############################################################################

class PathomicNet_Original(nn.Module):
    def __init__(self, args, act):
        super(PathomicNet_Original, self).__init__()
        init_max = True if args.init_type == "max" else False
        self.args = args
        self.act = act

        # self.path_net = ABMIL(args)
        # self.path_net = TransMIL(args)
        self.path_net = nn.Sequential(
            # nn.MaxPool1d(kernel_size=2500),
            nn.Linear(1024, self.args.path_dim)
            )
        self.path_classifier = nn.Sequential(
            nn.Linear(1024, self.args.label_dim),
            # nn.Sigmoid()
        )
        
        self.omic_net = MaxNet(input_dim=args.input_size_omic, omic_dim=args.omic_dim, return_grad = args.return_grad,
                               dropout_rate=args.dropout_rate, label_dim=args.label_dim, init_max=init_max)
        self.bilinear_dim = 20

        if args.fusion_type != "concat" and args.fusion_type != "add":
            self.fusion = define_bifusion(fusion_type=args.fusion_type, skip=args.skip, 
                            use_bilinear=args.use_bilinear, gate1=args.path_gate, 
                            gate2=args.omic_gate, dim1=args.path_dim, dim2=args.omic_dim, 
                            scale_dim1=args.path_scale, scale_dim2=args.omic_scale, 
                            mmhid=args.mmhid, dropout_rate=args.dropout_rate)
        
            self.classifier = nn.Sequential(nn.Linear(args.mmhid, args.label_dim))
        elif args.fusion_type == "add":
            self.classifier = nn.Sequential(nn.Linear(args.mmhid, args.label_dim))
        else:
            self.classifier = nn.Sequential(nn.Linear(2 * args.mmhid, args.label_dim))
            
        self.return_grad = args.return_grad
        self.cut_fuse_grad = args.cut_fuse_grad
        self.fusion_type = args.fusion_type

        # dfs_freeze(self.omic_net)
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, **kwargs):
        # path_vec = kwargs['x_path']
        # path_vec_f3, path_vec, hazard_path, pred_path, path_grads = self.path_net(x_path=kwargs['x_path'])
        # path_vec, hazard_path, path_grads = self.path_net(kwargs['x_path'])  # (BS,2500,1024), x_path x pathology
        x_path_input = torch.mean(kwargs['x_path'], dim=1)
        path_vec = self.path_net(x_path_input)
        hazard_path = self.path_classifier(kwargs['x_path'])
        omic_vec, hazard_omic, omic_grads = self.omic_net(x_omic=kwargs['x_omic'])
        #  kwargs['x_omic_tumor'], kwargs['x_omic_immune']
        # add one multi head attention with path_vec and omic_vec as the input features. The omic_vec should be query, and k, v comes from path_vec. 
        if self.cut_fuse_grad:
            if self.fusion_type == "concat":
                features = torch.cat((path_vec.clone().detach(), omic_vec.clone().detach()), 1)
            elif self.fusion_type == "add":
                features = torch.add(path_vec.clone().detach(), omic_vec.clone().detach())
            else:
                features = self.fusion(path_vec.clone().detach(), omic_vec.clone().detach())
        else:
            if self.fusion_type == "concat":
                # print('path_vec.shape:', path_vec.shape) 1,128
                # print('omic_vec.shape:', omic_vec.shape) 2,128
                features = torch.cat((path_vec, omic_vec), 1)
            elif self.fusion_type == "add":
                features = torch.add(path_vec, omic_vec)
            else:
                # print('path_vec.shape:', path_vec.shape)
                # print('omic_vec.shape:', omic_vec.shape)
                features = self.fusion(path_vec, omic_vec)
        
        # print("features for different branches:", path_vec.shape, omic_vec.shape, features.shape)
        hazard = self.classifier(features)
        # print("predictions:", F.softmax(hazard, dim=1))

        if self.return_grad == "True":
            fuse_grads = get_grad_embedding(hazard, features).detach().cpu().numpy()
        else:
            fuse_grads = None
        # print(fuse_grads.shape)
        # print(torch.sum(torch.abs(fuse_grads), axis=1))

        ### logits for the three branches.
        logits = [hazard_path, hazard_omic, hazard]

        return 0, 0, 0, logits, 0, 0, 0

    def __hasattr__(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return True
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return True
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return True
        return False

class PathomicNet(nn.Module):
    def __init__(self, args, act):
        super(PathomicNet, self).__init__()
        init_max = True if args.init_type == "max" else False
        self.args = args
        self.act = act

        self.path_net = ABMIL(args)
        # self.path_net = TransMIL(args)
        
        self.omic_net = MaxNet(input_dim=args.input_size_omic, omic_dim=args.omic_dim, return_grad = args.return_grad,
                               dropout_rate=args.dropout_rate, label_dim=args.label_dim, init_max=init_max)
        self.bilinear_dim = 20

        if args.fusion_type != "concat" and args.fusion_type != "add":
            self.fusion = define_bifusion(fusion_type=args.fusion_type, skip=args.skip, 
                            use_bilinear=args.use_bilinear, gate1=args.path_gate, 
                            gate2=args.omic_gate, dim1=args.path_dim, dim2=args.omic_dim, 
                            scale_dim1=args.path_scale, scale_dim2=args.omic_scale, 
                            mmhid=args.mmhid, dropout_rate=args.dropout_rate)
        
            self.classifier = nn.Sequential(nn.Linear(args.mmhid, args.label_dim))
        elif args.fusion_type == "add":
            self.classifier = nn.Sequential(nn.Linear(args.mmhid, args.label_dim))
        else:
            self.classifier = nn.Sequential(nn.Linear(2 * args.mmhid, args.label_dim))
            
        self.return_grad = args.return_grad
        self.cut_fuse_grad = args.cut_fuse_grad
        self.fusion_type = args.fusion_type

        # dfs_freeze(self.omic_net)
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, **kwargs):
        # path_vec = kwargs['x_path']
        # path_vec_f3, path_vec, hazard_path, pred_path, path_grads = self.path_net(x_path=kwargs['x_path'])
        path_vec, logits_path, path_grads = self.path_net(kwargs['x_path'])  # (BS,2500,1024), x_path x pathology
        
        omic_vec, logits_omic, omic_grads = self.omic_net(x_omic=kwargs['x_omic'])
        #  kwargs['x_omic_tumor'], kwargs['x_omic_immune']
        # print("path feature:", torch.mean(path_vec), torch.max(path_vec))
        # print("omic feature:", torch.mean(omic_vec), torch.max(omic_vec))
        
        # add one multi head attention with path_vec and omic_vec as the input features. The omic_vec should be query, and k, v comes from path_vec. 

        if self.cut_fuse_grad:
            if self.fusion_type == "concat":
                features = torch.cat((path_vec.clone().detach(), omic_vec.clone().detach()), 1)
            elif self.fusion_type == "add":
                features = torch.add(path_vec.clone().detach(), omic_vec.clone().detach())
            else:
                features = self.fusion(path_vec.clone().detach(), omic_vec.clone().detach())
        else:
            if self.fusion_type == "concat":
                # print('path_vec.shape:', path_vec.shape) 1,128
                # print('omic_vec.shape:', omic_vec.shape) 2,128
                features = torch.cat((path_vec, omic_vec), 1)
            elif self.fusion_type == "add":
                features = torch.add(path_vec, omic_vec)
            else:
                features = self.fusion(path_vec, omic_vec)
        
        # print("features for different branches:", path_vec.shape, omic_vec.shape, features.shape)
        logits_final = self.classifier(features)
        # print("predictions:", F.softmax(hazard, dim=1))
        ### logits for the three branches.
        logits = [logits_path, logits_omic, logits_final]
        return features, path_vec, omic_vec, logits, None, path_grads, omic_grads
        # return 0, 0, 0, logits, 0, 0, 0

    def __hasattr__(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return True
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return True
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return True
        return False

class DeformPathomicNet(nn.Module):
    def __init__(self, args):
        super(DeformPathomicNet, self).__init__()
        init_max = True if args.init_type == "max" else False
        
        # self.path_net = ABMIL(args)
        # self.path_net = TransMIL(args)
        self.args = args
        
        self.omic_net_tumor = MaxNet(input_dim=args.input_size_omic_tumor, omic_dim=args.omic_dim, return_grad = args.return_grad,
                               dropout_rate=args.dropout_rate, label_dim=args.label_dim, init_max=init_max)
        self.omic_net_immune = MaxNet(input_dim=args.input_size_omic_immune, omic_dim=args.omic_dim, return_grad = args.return_grad,
                               dropout_rate=args.dropout_rate, label_dim=args.label_dim, init_max=init_max)
        self.pathomic_net_tumor = DeformCrossTransMIL(args)
        self.pathomic_net_immune = DeformCrossTransMIL(args)
        
        self.bilinear_dim = 20

        if args.fusion_type != "concat":
            self.fusion = define_bifusion(fusion_type=args.fusion_type, skip=args.skip, 
                            use_bilinear=args.use_bilinear, gate1=args.path_gate, 
                            gate2=args.omic_gate, dim1=args.path_dim, dim2=args.omic_dim, 
                            scale_dim1=args.path_scale, scale_dim2=args.omic_scale, 
                            mmhid=args.mmhid, dropout_rate=args.dropout_rate)
        
            self.classifier = nn.Sequential(nn.Linear(args.mmhid, args.label_dim))
        else:
            # self.classifier = nn.Sequential(nn.Linear(args.mmhid * 2, args.label_dim))
            self.classifier = nn.Linear(args.mmhid * 2, args.label_dim)
                
            self.classifier_tumor = nn.Sequential(nn.Linear(args.mmhid, args.label_dim))
            self.classifier_immune = nn.Sequential(nn.Linear(args.mmhid, args.label_dim))
                    
        self.return_grad = args.return_grad
        self.cut_fuse_grad = args.cut_fuse_grad
        self.fusion_type = args.fusion_type

        # dfs_freeze(self.omic_net)
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, **kwargs):
        # path_vec = kwargs['x_path']
        # path_vec_f3, path_vec, hazard_path, pred_path, path_grads = self.path_net(x_path=kwargs['x_path'])
        # path_vec, hazard_path, path_grads = self.path_net(kwargs['x_path'])  # (BS,2500,1024), x_path x pathology
        # print('x_omic_tumor.shape:', kwargs['x_omic_tumor'].shape)
        if not self.args.return_vgrid:
            omic_vec_tumor, hazard_omic, omic_grads = self.omic_net_tumor(x_omic=kwargs['x_omic_tumor']) #input: B,59
            pathomic_vec_tumor, hazard_pathomic_tumor, pathomic_grads_tumor = self.pathomic_net_tumor(path=kwargs['x_path'], omic=omic_vec_tumor)
            
            omic_vec_immune, hazard_omic, omic_grads = self.omic_net_immune(x_omic=kwargs['x_omic_immune'])
            pathomic_vec_immune, hazard_pathomic_immune, pathomic_grads_immune = self.pathomic_net_immune(path=kwargs['x_path'], omic=omic_vec_immune)
        elif self.args.return_vgrid:
            omic_vec_tumor, hazard_omic, omic_grads = self.omic_net_tumor(x_omic=kwargs['x_omic_tumor']) #input: B,59
            pathomic_vec_tumor, hazard_pathomic_tumor, pathomic_grads_tumor, omic_tumor, vgrid_tumor = self.pathomic_net_tumor(path=kwargs['x_path'], omic=omic_vec_tumor)
            
            omic_vec_immune, hazard_omic, omic_grads = self.omic_net_immune(x_omic=kwargs['x_omic_immune'])
            pathomic_vec_immune, hazard_pathomic_immune, pathomic_grads_immune, omic_immune, vgrid_immune = self.pathomic_net_immune(path=kwargs['x_path'], omic=omic_vec_immune)

        # kwargs['x_omic_tumor'], kwargs['x_omic_immune']
        # print("path feature:", torch.mean(path_vec), torch.max(path_vec))
        # print("omic feature:", torch.mean(omic_vec), torch.max(omic_vec))
        
        # add one multi head attention with path_vec and omic_vec as the input features. The omic_vec should be query, and k, v comes from path_vec. 

        if self.cut_fuse_grad:
            if self.fusion_type == "concat":
                features = torch.cat((pathomic_vec_tumor.clone().detach(), pathomic_vec_immune.clone().detach()), 1)
            else:
                features = self.fusion(pathomic_vec_tumor.clone().detach(), pathomic_vec_immune.clone().detach())
        else:
            if self.fusion_type == "concat":
                # print('path_vec.shape:', path_vec.shape) 1,128
                # print('omic_vec.shape:', omic_vec.shape) 2,128
                features = torch.cat((pathomic_vec_tumor, pathomic_vec_immune), 1)
            else:
                features = self.fusion(pathomic_vec_tumor, pathomic_vec_immune)
        
        # print("features for different branches:", path_vec.shape, omic_vec.shape, features.shape)

        hazard = self.classifier(features)
        # print("predictions:", F.softmax(hazard, dim=1))
        hazard_tumor = self.classifier_tumor(pathomic_vec_tumor)
        hazard_immune = self.classifier_immune(pathomic_vec_immune)

        if self.return_grad == "True":
            fuse_grads = get_grad_embedding(hazard, features).detach().cpu().numpy()
        else:
            fuse_grads = None
        # print(fuse_grads.shape)
        # print(torch.sum(torch.abs(fuse_grads), axis=1))
        if self.args.task_type=="survival":
            hazard = torch.sigmoid(hazard)
            hazard_tumor = torch.sigmoid(hazard_tumor)
            hazard_immune = torch.sigmoid(hazard_immune)
            # if isinstance(self.act, nn.Sigmoid):
            #     hazard = hazard * self.output_range + self.output_shift
                
        ### logits for the three branches.
        if self.args.return_vgrid:
            logits = [hazard_tumor, hazard_immune, hazard, omic_tumor, vgrid_tumor, omic_immune, vgrid_immune]
        else:
            logits = [hazard_tumor, hazard_immune, hazard]
        
        return features, pathomic_vec_tumor, pathomic_vec_immune, logits, fuse_grads, pathomic_grads_tumor, pathomic_grads_immune


from .mcat_utils import Attn_Net_Gated, SNN_Block, MCAT_BilinearFusion
# from .mcat_utils import SNN_Block, MultiheadAttention, Attn_Net_Gated, MCAT_BilinearFusion
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

###########################
### MCAT Implementation ###
###########################
#[100, 100, 100, 131] or [431]
class MCAT_Surv(nn.Module):
    def __init__(self, args, fusion='concat', omic_sizes=[100, 100, 100, 131], n_classes=4,
                 model_size_wsi: str='small', model_size_omic: str='small', dropout=0.25):
        super(MCAT_Surv, self).__init__()
        self.args = args
        self.fusion = fusion
        self.omic_sizes = omic_sizes
        self.n_classes = args.label_dim
        self.size_dict_WSI = {"small": [1024, 256, 256], "big": [1024, 512, 384]}
        self.size_dict_omic = {'small': [256, 256], 'big': [1024, 1024, 1024, 256]}
        
        ### FC Layer over WSI bag
        size = self.size_dict_WSI[model_size_wsi]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        fc.append(nn.Dropout(0.25))
        self.wsi_net = nn.Sequential(*fc)
        
        ### Constructing Genomic SNN
        hidden = self.size_dict_omic[model_size_omic]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.sig_networks = nn.ModuleList(sig_networks)

        ### Multihead Attention
        self.coattn = MultiheadAttention(embed_dim=256, num_heads=1)

        ### Path Transformer + Attention Head
        path_encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=dropout, activation='relu')
        self.path_transformer = nn.TransformerEncoder(path_encoder_layer, num_layers=2)
        self.path_attention_head = Attn_Net_Gated(L=size[2], D=size[2], dropout=dropout, n_classes=1)
        self.path_rho = nn.Sequential(*[nn.Linear(size[2], size[2]), nn.ReLU(), nn.Dropout(dropout)])
        
        ### Omic Transformer + Attention Head
        omic_encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=dropout, activation='relu')
        self.omic_transformer = nn.TransformerEncoder(omic_encoder_layer, num_layers=2)
        self.omic_attention_head = Attn_Net_Gated(L=size[2], D=size[2], dropout=dropout, n_classes=1)
        self.omic_rho = nn.Sequential(*[nn.Linear(size[2], size[2]), nn.ReLU(), nn.Dropout(dropout)])
        
        ### Fusion Layer
        if self.fusion == 'concat':
            self.mm = nn.Sequential(*[nn.Linear(256*2, size[2]), nn.ReLU(), nn.Linear(size[2], size[2]), nn.ReLU()])
        elif self.fusion == 'bilinear':
            self.mm = BilinearFusion(dim1=256, dim2=256, scale_dim1=8, scale_dim2=8, mmhid=256)
        else:
            self.mm = None
        
        ### Classifier
        self.classifier = nn.Linear(size[2], self.n_classes)

    def forward(self, **kwargs):
        x_path = kwargs['x_path']
        # x_omic = [kwargs['x_omic%d' % i] for i in range(1,7)]
        x_omic_all = kwargs['x_omic']
        # print('x_path, x_omic_all.shape:', x_path.shape, x_omic_all.shape)
        omic_sizes = self.omic_sizes
        x_omic =  [x_omic_all[:, sum(omic_sizes[:i]):sum(omic_sizes[:i+1])] for i in range(len(omic_sizes))]


        h_path_bag = self.wsi_net(x_path).transpose(0,1) ### path embeddings are fed through a FC layer
        h_omic = [self.sig_networks[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)] ### each omic signature goes through it's own FC layer
        h_omic_bag = torch.stack(h_omic) ### omic embeddings are stacked (to be used in co-attention)

        # Coattn
        # print('shape of inputs are:', h_omic_bag.shape, h_path_bag.shape) #([4, 8, 256]), [2500, 8, 256]
        h_path_coattn, A_coattn = self.coattn(h_omic_bag, h_path_bag, h_path_bag)
        # print('shape of h_path_coattn, A_coattn:', h_path_coattn.shape, A_coattn.shape) #[4, 8, 256], [8, 1, 4, 2500]

        ### Path
        h_path_trans = self.path_transformer(h_path_coattn)
        # print('h_path_trans.shape:', h_path_trans.shape) #torch.Size([4, 8, 256])
        A_path, h_path = self.path_attention_head(h_path_trans.squeeze(1))
        # print('A_path, h_path.shape:', A_path.shape, h_path.shape) #([4, 8, 1],[4, 8, 256]
        A_path = torch.transpose(A_path.transpose(0,2), 1, 0)
        h_path = h_path.transpose(0,1)
        # print('A_path.shape:', A_path.shape) #[8,4] -> 8,1,4
        # print('h_path.shape:', h_path.shape) # ->8,4,256
        h_path = torch.bmm(F.softmax(A_path, dim=2).float(), h_path.float())
        h_path = self.path_rho(h_path).squeeze()
        # print('h_path.shape:', h_path.shape) #8,1,256 -> 8,256
        
        ### Omic
        h_omic_trans = self.omic_transformer(h_omic_bag)
        A_omic, h_omic = self.omic_attention_head(h_omic_trans.squeeze(1))
        A_omic = torch.transpose(A_omic.transpose(0,2), 1, 0)
        h_omic = h_omic.transpose(0,1)
        # print('A_omic.shape:', A_omic.shape) # ->8,1,4
        # print('h_omic.shape:', h_omic.shape) # ->8,4,256
        h_omic = torch.bmm(F.softmax(A_omic, dim=2).float(), h_omic.float())
        h_omic = self.omic_rho(h_omic).squeeze()
        # print('h_omic.shape:', h_omic.shape) #8,256
        
        if self.fusion == 'bilinear':
            h = self.mm(h_path.unsqueeze(dim=0), h_omic.unsqueeze(dim=0)).squeeze()
        elif self.fusion == 'concat':
            h = self.mm(torch.cat([h_path, h_omic], axis=1))
                
        ### Survival Layer
        logits = self.classifier(h)
        # print('logits.shape:', logits.shape) # [8,4]
        # Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        attention_scores = {'coattn': A_coattn, 'path': A_path, 'omic': A_omic}
        return logits, hazards, S


    def captum(self, x_path, x_omic1, x_omic2, x_omic3, x_omic4, x_omic5, x_omic6):
        #x_path = torch.randn((10, 500, 1024))
        #x_omic1, x_omic2, x_omic3, x_omic4, x_omic5, x_omic6 = [torch.randn(10, size) for size in omic_sizes]
        x_omic = [x_omic1, x_omic2, x_omic3, x_omic4, x_omic5, x_omic6]
        h_path_bag = self.wsi_net(x_path)#.unsqueeze(1) ### path embeddings are fed through a FC layer
        h_path_bag = torch.reshape(h_path_bag, (500, 10, 256))
        h_omic = [self.sig_networks[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)] ### each omic signature goes through it's own FC layer
        h_omic_bag = torch.stack(h_omic) ### omic embeddings are stacked (to be used in co-attention)

        # Coattn
        h_path_coattn, A_coattn = self.coattn(h_omic_bag, h_path_bag, h_path_bag)

        ### Path
        h_path_trans = self.path_transformer(h_path_coattn)
        h_path_trans = torch.reshape(h_path_trans, (10, 6, 256))
        A_path, h_path = self.path_attention_head(h_path_trans)
        A_path = F.softmax(A_path.squeeze(dim=2), dim=1).unsqueeze(dim=1)
        h_path = torch.bmm(A_path, h_path).squeeze(dim=1)

        ### Omic
        h_omic_trans = self.omic_transformer(h_omic_bag)
        h_omic_trans = torch.reshape(h_omic_trans, (10, 6, 256))
        A_omic, h_omic = self.omic_attention_head(h_omic_trans)
        A_omic = F.softmax(A_omic.squeeze(dim=2), dim=1).unsqueeze(dim=1)
        h_omic = torch.bmm(A_omic, h_omic).squeeze(dim=1)

        if self.fusion == 'bilinear':
            h = self.mm(h_path.unsqueeze(dim=0), h_omic.unsqueeze(dim=0)).squeeze()
        elif self.fusion == 'concat':
            h = self.mm(torch.cat([h_path, h_omic], axis=1))

        logits  = self.classifier(h)
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)

        risk = -torch.sum(S, dim=1)
        return risk


###########################
### CMTA Implementation ###
###########################
from .cmta_utils import Transformer_P, Transformer_G

#[100, 100, 100, 131] or [431]
class CMTA(nn.Module):
    def __init__(self, args, fusion='concat', omic_sizes=[100, 100, 100, 131], n_classes=4,
                 model_size_wsi: str='small', model_size_omic: str='small', dropout=0.25):
        super(CMTA, self).__init__()
        self.args = args
        self.fusion = fusion
        self.omic_sizes = omic_sizes
        self.n_classes = args.label_dim
        self.size_dict_WSI = {"small": [1024, 256, 256], "big": [1024, 512, 384]}
        self.size_dict_omic = {'small': [256, 256], 'big': [1024, 1024, 1024, 256]}
        
        ### FC Layer over WSI bag
        size = self.size_dict_WSI[model_size_wsi]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        fc.append(nn.Dropout(0.25))
        self.wsi_net = nn.Sequential(*fc)
        
        ### Constructing Genomic SNN
        hidden = self.size_dict_omic[model_size_omic]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.sig_networks = nn.ModuleList(sig_networks)

        # Pathomics Transformer
        # Encoder
        self.pathomics_encoder = Transformer_P(feature_dim=hidden[-1])
        # Decoder
        self.pathomics_decoder = Transformer_P(feature_dim=hidden[-1])

        # P->G Attention
        self.P_in_G_Att = MultiheadAttention(embed_dim=256, num_heads=1)
        # G->P Attention
        self.G_in_P_Att = MultiheadAttention(embed_dim=256, num_heads=1)

        # Pathomics Transformer Decoder
        # Encoder
        self.genomics_encoder = Transformer_G(feature_dim=hidden[-1])
        # Decoder
        self.genomics_decoder = Transformer_G(feature_dim=hidden[-1])

        # ### Multihead Attention
        # self.coattn = MultiheadAttention(embed_dim=256, num_heads=1)

        # ### Path Transformer + Attention Head
        # path_encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=dropout, activation='relu')
        # self.path_transformer = nn.TransformerEncoder(path_encoder_layer, num_layers=2)
        # self.path_attention_head = Attn_Net_Gated(L=size[2], D=size[2], dropout=dropout, n_classes=1)
        # self.path_rho = nn.Sequential(*[nn.Linear(size[2], size[2]), nn.ReLU(), nn.Dropout(dropout)])
        
        # ### Omic Transformer + Attention Head
        # omic_encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=dropout, activation='relu')
        # self.omic_transformer = nn.TransformerEncoder(omic_encoder_layer, num_layers=2)
        # self.omic_attention_head = Attn_Net_Gated(L=size[2], D=size[2], dropout=dropout, n_classes=1)
        # self.omic_rho = nn.Sequential(*[nn.Linear(size[2], size[2]), nn.ReLU(), nn.Dropout(dropout)])
        
        ### Fusion Layer
        if self.fusion == 'concat':
            self.mm = nn.Sequential(*[nn.Linear(256*2, size[2]), nn.ReLU(), nn.Linear(size[2], size[2]), nn.ReLU()])
        elif self.fusion == 'bilinear':
            self.mm = BilinearFusion(dim1=256, dim2=256, scale_dim1=8, scale_dim2=8, mmhid=256)
        else:
            self.mm = None
        
        ### Classifier
        self.classifier = nn.Linear(size[2], self.n_classes)

    def forward(self, **kwargs):
        x_path = kwargs['x_path']
        # x_omic = [kwargs['x_omic%d' % i] for i in range(1,7)]
        x_omic_all = kwargs['x_omic']
        # print('x_path, x_omic_all.shape:', x_path.shape, x_omic_all.shape)
        omic_sizes = self.omic_sizes
        x_omic =  [x_omic_all[:, sum(omic_sizes[:i]):sum(omic_sizes[:i+1])] for i in range(len(omic_sizes))]


        h_path_bag = self.wsi_net(x_path) ### path embeddings are fed through a FC layer
        h_omic = [self.sig_networks[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)] ### each omic signature goes through it's own FC layer
        h_omic_bag = torch.stack(h_omic).transpose(0,1) ### omic embeddings are stacked (to be used in co-attention)
        genomics_features = h_omic_bag #[8, 4, 256]
        pathomics_features = h_path_bag #[8, 2500, 256]


        # encoder
        # pathomics encoder
        cls_token_pathomics_encoder, patch_token_pathomics_encoder = self.pathomics_encoder(
            pathomics_features)  # cls token + patch tokens
        # genomics encoder
        cls_token_genomics_encoder, patch_token_genomics_encoder = self.genomics_encoder(
            genomics_features)  # cls token + patch tokens

        # cross-omics attention
        pathomics_in_genomics, Att = self.P_in_G_Att(
            patch_token_pathomics_encoder.transpose(1, 0),
            patch_token_genomics_encoder.transpose(1, 0),
            patch_token_genomics_encoder.transpose(1, 0),
        )  # ([14642, 1, 256])
        genomics_in_pathomics, Att = self.G_in_P_Att(
            patch_token_genomics_encoder.transpose(1, 0),
            patch_token_pathomics_encoder.transpose(1, 0),
            patch_token_pathomics_encoder.transpose(1, 0),
        )  # ([7, 1, 256])
        # decoder
        # pathomics decoder
        cls_token_pathomics_decoder, _ = self.pathomics_decoder(
            pathomics_in_genomics.transpose(1, 0))  # cls token + patch tokens
        # genomics decoder
        cls_token_genomics_decoder, _ = self.genomics_decoder(
            genomics_in_pathomics.transpose(1, 0))  # cls token + patch tokens

        # fusion
        if self.fusion == "concat":
            fusion = self.mm(
                torch.concat(
                    (
                        (cls_token_pathomics_encoder + cls_token_pathomics_decoder) / 2,
                        (cls_token_genomics_encoder + cls_token_genomics_decoder) / 2,
                    ),
                    dim=1,
                )
            )  # take cls token to make prediction
        elif self.fusion == "bilinear":
            fusion = self.mm(
                (cls_token_pathomics_encoder + cls_token_pathomics_decoder) / 2,
                (cls_token_genomics_encoder + cls_token_genomics_decoder) / 2,
            )  # take cls token to make prediction
        else:
            raise NotImplementedError("Fusion [{}] is not implemented".format(self.fusion))

        # predict
        # print('fusion.shape:', fusion.shape) #8,256
        logits = self.classifier(fusion)  # [1, n_classes]
        # print('logits.shape:', logits.shape) #8,4
        hazards = torch.sigmoid(logits)
        # print('hazards.shape:', hazards.shape) #8,4
        S = torch.cumprod(1 - hazards, dim=1)
        return logits, hazards, S, cls_token_pathomics_encoder, cls_token_pathomics_decoder, cls_token_genomics_encoder, cls_token_genomics_decoder

                
        # ### Survival Layer 
        # logits = self.classifier(h).unsqueeze(0)
        # Y_hat = torch.topk(logits, 1, dim = 1)[1]
        # hazards = torch.sigmoid(logits)
        # S = torch.cumprod(1 - hazards, dim=1)
        
        # attention_scores = {'coattn': A_coattn, 'path': A_path, 'omic': A_omic}
        
        # hazards = hazards.squeeze(0)
        # # print('hazards.shape:', hazards.shape) #8,4
        # # return hazards, S, Y_hat, attention_scores
        # return hazards



        # # Coattn
        # # print('shape of inputs are:', h_omic_bag.shape, h_path_bag.shape) #([4, 8, 256]), [2500, 8, 256]
        # h_path_coattn, A_coattn = self.coattn(h_omic_bag, h_path_bag, h_path_bag)
        # # print('shape of h_path_coattn, A_coattn:', h_path_coattn.shape, A_coattn.shape) #[4, 8, 256], [8, 1, 4, 2500]

        # ### Path
        # h_path_trans = self.path_transformer(h_path_coattn)
        # # print('h_path_trans.shape:', h_path_trans.shape) #torch.Size([4, 8, 256])
        # A_path, h_path = self.path_attention_head(h_path_trans.squeeze(1))
        # # print('A_path, h_path.shape:', A_path.shape, h_path.shape) #([4, 8, 1],[4, 8, 256]
        # A_path = torch.transpose(A_path.transpose(0,2), 1, 0)
        # h_path = h_path.transpose(0,1)
        # # print('A_path.shape:', A_path.shape) #[8,4] -> 8,1,4
        # # print('h_path.shape:', h_path.shape) # ->8,4,256
        # h_path = torch.bmm(F.softmax(A_path, dim=2).float(), h_path.float())
        # h_path = self.path_rho(h_path).squeeze()
        # # print('h_path.shape:', h_path.shape) #8,1,256 -> 8,256
        
        # ### Omic
        # h_omic_trans = self.omic_transformer(h_omic_bag)
        # A_omic, h_omic = self.omic_attention_head(h_omic_trans.squeeze(1))
        # A_omic = torch.transpose(A_omic.transpose(0,2), 1, 0)
        # h_omic = h_omic.transpose(0,1)
        # # print('A_omic.shape:', A_omic.shape) # ->8,1,4
        # # print('h_omic.shape:', h_omic.shape) # ->8,4,256
        # h_omic = torch.bmm(F.softmax(A_omic, dim=2).float(), h_omic.float())
        # h_omic = self.omic_rho(h_omic).squeeze()
        # # print('h_omic.shape:', h_omic.shape) #8,256
