import torch
import torch.nn as nn
import torch.nn.functional as F
from .gather import GatherLayer
from torch.autograd import Variable      
    
class BatchLoss(nn.Module):
    def __init__(self, batch_size, world_size):
        super(BatchLoss, self).__init__()
        self.batch_size = batch_size
        self.world_size = world_size

    def forward(self, omic, vgrid):
        # assert omic.size() == vgrid.size()
        # omic.shape [B, 128], vgrid.shape[8B, 2, 12, 12]
        N = self.batch_size * self.world_size
        # gather data from all the processes
        if self.world_size > 1:
            omic = torch.cat(GatherLayer.apply(omic), dim=0)
            vgrid = torch.cat(GatherLayer.apply(vgrid), dim=0)
        # reshape as N*C
        omic = omic.view(N, -1)
        vgrid = vgrid.view(8, N, -1)

        # form N*N similarity matrix
        similarity = omic.mm(omic.t())
        norm = torch.norm(similarity, 2, 1).view(-1, 1)
        similarity = similarity / norm

        vgrid_similarities = []
        for item in vgrid:
            vgrid_similarity = item.mm(item.t())
            vgrid_norm = torch.norm(vgrid_similarity, 2, 1).view(-1, 1)
            vgrid_similarity = vgrid_similarity / vgrid_norm
            vgrid_similarities.append(vgrid_similarity)
        mean_vgrid_sim = torch.mean(torch.stack(vgrid_similarities), dim=0)

        batch_loss = (similarity - mean_vgrid_sim) ** 2 / N
        
        return batch_loss
