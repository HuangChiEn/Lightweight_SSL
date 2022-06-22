import torch
import torch.nn.functional as F
from torch import nn
from util_tool.utils import get_rank, dist_gather

class NearestNeighborLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, nn, p):
        """Computes NNCLR's loss given batch of nearest-neighbors nn from view 1 and
        predicted features p from view 2.
        Args:
            nn (torch.Tensor): NxD Tensor containing nearest neighbors' features from view 1.
            p (torch.Tensor): NxD Tensor containing predicted features from view 2
            temperature (float, optional): temperature of the softmax in the contrastive loss. Defaults
                to 0.1.
        Returns:
            torch.Tensor: NNCLR loss.
        """
        nn = F.normalize(nn, dim=-1)
        p = F.normalize(p, dim=-1)
        # to be consistent with simclr, we now gather p
        # this might result in suboptimal results given previous parameters.
        p = dist_gather(p)

        logits = nn @ p.T / self.temperature

        rank = get_rank()
        n = nn.size(0)
        labels = torch.arange(n * rank, n * (rank + 1), device=p.device)
        loss = F.cross_entropy(logits, labels)
        return loss