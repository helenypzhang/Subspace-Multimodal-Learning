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


"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    K: number of attention maps, number of attention heads
"""
class ABMIL(nn.Module):
    def __init__(self, args):
        super(ABMIL, self).__init__()
        self.L = 1024
        self.D = 128
        self.K = 1
        self.args = args
        self.n_classes = args.label_dim
        
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D), # B,N,128
            nn.Tanh(),
            nn.Linear(self.D, self.K) # B,N,K
        )
        

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, self.n_classes),
            # nn.Sigmoid()
        )
        
        # multi-modal projection
        # self.multimodal_projection = nn.Sequential(
        #     nn.Linear(self.L*self.K, self.args.path_dim),
        #     nn.BatchNorm1d(self.args.path_dim),
        #     nn.ReLU(inplace=True)
        # ) 
        self.multimodal_projection = nn.Linear(self.L*self.K, self.args.path_dim)

    def forward(self, x):
        # x = x.view(-1, self.args.input_path_dim)  # N x input_path_dim
        
        A = self.attention(x)  # NxK [B,N,K]
        
        # A = torch.transpose(A, 1, 0)  # KxN  [B,K,N]
        # A = F.softmax(A, dim=1)  # softmax over N [B,K,N]
        # M = torch.mm(A, x)  # KxL [B,K,L]
        
        A = torch.transpose(A, 2, 1) #[B,K,N]
        A = F.softmax(A, dim=2) # softmax over N; A:[B,K,N], x:[B,N,L]
        M = torch.bmm(A, x) #[B,K,L]
        M = M.view(-1, self.L*self.K) #[B, K*L]


        logits = self.classifier(M)
        
        encoded = self.multimodal_projection(M) # Bx128
        
        return encoded, logits, None #[BS, cls]
    
    
    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A


class GatedABMIL(nn.Module):
    def __init__(self):
        super(GatedABMIL, self).__init__()
        self.L = 1024
        self.D = 128
        self.K = 1


        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 2),
            nn.Sigmoid()
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, subtype_labels, adj, mask):
        x = x.squeeze(0)

        H = x

        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        prob = self.classifier(M)
        _, pred = torch.max(prob, 1)

        # prob
        subtype_prob = prob.data
        # pred
        subtype_pred = pred

        subtye_loss = self.criterion(prob, subtype_labels)

        return subtype_prob, subtype_pred, subtype_labels, subtye_loss

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A


class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, args):
        super(TransMIL, self).__init__()
        self.args = args
        self.pos_layer = PPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.n_classes = self.args.label_dim
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.n_classes)

        # multi-modal projection
        self.multimodal_projection = nn.Linear(512, self.args.path_dim)

    def forward(self, x):

        h = x.float() #[B, n, 1024]
        
        h = self._fc1(h) #[B, n, 512]
        
        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer x1
        h = self.layer1(h) #[B, N, 512]

        #---->PPEG
        h = self.pos_layer(h, _H, _W) #[B, N, 512] [B,2500,512]
        
        #---->Translayer x2
        h = self.layer2(h) #[B, N, 512]

        #---->cls_token
        h = self.norm(h)[:,0]

        #---->predict
        logits = self._fc2(h) #[B, n_classes]
        
        encoded = self.multimodal_projection(h) # Bx128
        
        return encoded, logits, None #[BS, cls]

# if __name__ == "__main__":
#     data = torch.randn((1, 6000, 1024)).cuda()
#     model = TransMIL(n_classes=2).cuda()
#     print(model.eval())
#     results_dict = model(data = data)
#     print(results_dict)