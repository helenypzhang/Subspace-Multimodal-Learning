import math
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.cuda
from torch.autograd import Variable
# import basic_net as basic_net
import yaml
import os
from yaml.loader import SafeLoader
import copy
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# import basic_net as basic_net
from torch.nn.modules.utils import _pair
from scipy import ndimage

from nystrom_attention import NystromAttention
from .DeformableAttention2D import DeformCrossAttention2D
from .DeformableAttention1D import DeformCrossAttention1D

class FusionNet(nn.Module):
    def __init__(self, feature_dim=128):
        super(FusionNet, self).__init__()
        self.fusion_layer = nn.Linear(feature_dim * 2, feature_dim)

    def forward(self, gene_features, image_features):
        # Concatenate features along the last dimension
        combined_features = torch.cat((gene_features, image_features), dim=-1)
        # Use a linear layer to learn the fusion
        fused_features = self.fusion_layer(combined_features)
        return fused_features

class DeformCrossTransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=128):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn2d = DeformCrossAttention2D(
            dim = 128,                   # feature dimensions
            dim_head = 64,               # dimension per head
            heads = 8,                   # attention heads
            dropout = 0.1,                # dropout
            downsample_factor = 4,       # downsample factor (r in paper)
            offset_scale = 4,            # scale of offset, maximum offset
            offset_groups = 8,        # number of offset groups, should be multiple of heads, original = None.
            offset_kernel_size = 6,      # offset kernel size
        )
        self.attn1d = DeformCrossAttention1D(
            dim = 128,
            downsample_factor = 4,
            offset_scale = 2,
            offset_kernel_size = 6
        )

    def forward(self, x1, x2, attn_dim, return_vgrid):
        # x1(1, 1+2500, 512), x2(1, 1+2500, 512)
        # x1 = x1 + self.attn(self.norm(x1))
        if attn_dim == 1:
            x = self.attn1d(self.norm(x1).transpose(1, 2), self.norm(x2).transpose(1, 2))
            x = x1 + x.transpose(1, 2) # x(1, 1+2500, 512)
            return x
        elif attn_dim == 2:
            if not return_vgrid:
                x = self.attn2d(self.norm(x1).transpose(1, 2), self.norm(x2).transpose(1, 2), return_vgrid)
                x = x1 + x.transpose(1, 2) # x(1, 2500, 512)
                return x
            elif return_vgrid:
                x, vgrid = self.attn2d(self.norm(x1).transpose(1, 2), self.norm(x2).transpose(1, 2), return_vgrid)
                x = x1 + x.transpose(1, 2) # x(1, 2500, 512)
                return x, vgrid

class DeformCrossTransMIL(nn.Module):
    def __init__(self, args, n_classes=4):
        super(DeformCrossTransMIL, self).__init__()
        self.fusion_layer = FusionNet(feature_dim=128)
        self._fc1 = nn.Sequential(nn.Linear(1024, args.path_dim), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, args.path_dim))
        self.args = args
        self.n_classes = n_classes
        # self.layer1 = TransLayer(dim=512)
        # self.layer2 = TransLayer(dim=512)
        self.layer3 = DeformCrossTransLayer(dim=args.path_dim)
        self.norm = nn.LayerNorm(args.path_dim)
        self._fc2 = nn.Linear(args.path_dim, self.n_classes)
        
        self.pooler = Pooler(args.path_dim)
        # multi-modal projection
        self.multimodal_projection = nn.Linear(args.path_dim, self.args.path_dim)
    
    def forward(self, path, omic):

        path = path.float() #[B, n, 1024]
        path = self._fc1(path) #[B, n, 512]
        
        omic = omic.float() 
        # print('omic.shape:', omic.shape) #[B, 128]
        omic = omic.unsqueeze(1).repeat(1, 2500, 1)
        # omic = omic.unsqueeze(1).expand(-1, 2500, -1) #[B, N, 512] [B,2500,512]
        # or linear = nn.Linear(512, 2500*512)
        # omic = linear(omic).view(-1, 2500, 512)

        #---->Fusion
        h = self.fusion_layer(path, omic) #[B, N, 512] [B,2500,512], h(1, 2500, 512)
        
        if self.args.attn_dim == 1:
            #---->cls_token for fusion feature
            B = h.shape[0]
            cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
            # for fusion feature
            # print('h.shape:', h.shape) #[32, 2500, 512]
            h = torch.cat((cls_tokens, h), dim=1)
            # for path feature #[32, 2500, 512]
            # print('path.shape:', path.shape)
            path = torch.cat((cls_tokens, path), dim=1)

            #---->Translayer x1
            h = self.layer3(h, path, self.args.attn_dim, self.args.return_vgrid) #[B, N, 512]

            #---->cls_token
            h = self.norm(h)[:,0]
            # print('h.shape', h.shape)

            #---->predict
            logits = self._fc2(h) #[B, n_classes]
            
        elif self.args.attn_dim == 2:
            #---->Translayer x1
            if self.args.return_vgrid:
                h, vgrid = self.layer3(h, path, self.args.attn_dim, self.args.return_vgrid) #[B, N, 512]
            else:
                h = self.layer3(h, path, self.args.attn_dim, self.args.return_vgrid) #[B, N, 512]
                
            # h = self.layer3(h, path, self.args.attn_dim) #[B, N, 512]

            #---->cls_token
            # h = self.norm(h)[:,0]
            h = self.pooler(self.norm(h)) #[B, 512]
            # print('h.shape', h.shape)

            #---->predict
            logits = self._fc2(h) #[B, n_classes]     
        
        encoded = self.multimodal_projection(h) # Bx128
        path_grads = None
        
        # Y_hat = torch.argmax(logits, dim=1)
        # Y_prob = F.softmax(logits, dim = 1)
        # results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        # return results_dict
        if self.args.return_vgrid:
            return encoded, logits, path_grads, omic, vgrid#[BS, cls]
        else:
            return encoded, logits, path_grads #[BS, cls]

# if __name__ == "__main__":
#     data = torch.randn((1, 6000, 1024)).cuda()
#     model = TransMIL(n_classes=2).cuda()
#     print(model.eval())
#     results_dict = model(data = data)
#     print(results_dict)

class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # torch.Size([90, 64, 768]) language #[B, N, 512]
        # torch.Size([90, 82, 768]) vision
        # first_token: torch.Size([90, 768])
        # pooled_output1: torch.Size([90, 768])
        # pooled_output2: torch.Size([90, 768])
        
        # method 1:
        # first_token_tensor = hidden_states[:, 0]
        # pooled_output = self.dense(first_token_tensor)
        # pooled_output = self.activation(pooled_output)

        # mean_token_tensor: torch.Size([90, 768])
        # pooled_output1: torch.Size([90, 768])
        # pooled_output2: torch.Size([90, 768])
        
        # method 2:
        # 2.1 calculate average embeddings
        avg_token_tensor = torch.mean(hidden_states, dim = 1) # avg_embeddings shape: (90, 768)
        # print('avg_token_tensor.shape:', avg_token_tensor.shape)

        # 2.2 calculate pooled embeddings
        # pooled_token_tensor = torch.max(hidden_states, dim=1)[0] # pooled_embeddings shape: (90, 768)

        pooled_output = self.dense(avg_token_tensor)
        pooled_output = self.activation(pooled_output)

        return pooled_output