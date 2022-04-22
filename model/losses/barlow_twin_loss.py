import torch
import torch.nn.functional as F
from torch import nn
from util_tool.utils import get_rank, dist_gather

class BarlowTwinLoss(nn.Module):
    def __init__(self, simplified=True):
        super().__init__()
        self.simplified = simplified

    def forward(self, p, z):
        if self.simplified:
            return -F.cosine_similarity(p, z.detach(), dim=-1).mean()

        p = F.normalize(p, dim=-1)
        z = F.normalize(z, dim=-1)

        return -(p * z.detach()).sum(dim=1).mean()