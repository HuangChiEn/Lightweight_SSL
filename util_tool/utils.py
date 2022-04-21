import logging
import torch
import torch.distributed as dist
from pytorch_lightning.loggers import WandbLogger

def get_wandb_logger(save_dir, name, project, entity, offline):
    wandb_logger = WandbLogger(
        save_dir=save_dir,
        name=name,
        project=project,
        entity=entity,
        offline=offline,
    )
    return wandb_logger


class GatherLayer(torch.autograd.Function):
    """
    Gathers tensors from all process and supports backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized():
            output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(output, x)
        else:
            output = [x]
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        if dist.is_available() and dist.is_initialized():
            all_gradients = torch.stack(grads)
            dist.all_reduce(all_gradients)
            grad_out = all_gradients[get_rank()]
        else:
            grad_out = grads[0]
        return 

def dist_gather(X, dim=0):
    """Gathers tensors from all processes, supporting backward propagation."""
    gather_tnsr = torch.cat(GatherLayer.apply(X), dim=dim)
    gather_tnsr = [gather_tnsr] if not isinstance(gather_tnsr, list) else gather_tnsr
    return torch.cat(gather_tnsr)


@torch.no_grad()
def accuracy_at_k(outputs, targets, top_k = (1, 5)):
    """Computes the accuracy over the k top predictions for the specified values of k.
    Args:
        outputs (torch.Tensor): output of a classifier (logits or probabilities).
        targets (torch.Tensor): ground truth labels.
        top_k (Sequence[int], optional): sequence of top k values to compute the accuracy over.
            Defaults to (1, 5).
    Returns:
        Sequence[int]:  accuracies at the desired k.
    """
    maxk = max(top_k)
    batch_size = targets.size(0)

    _, pred = outputs.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    res = []
    for k in top_k:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def device_as(t1, t2):
   """
   Moves t1 to the device of t2
   """
   return t1.to(t2.device)

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count