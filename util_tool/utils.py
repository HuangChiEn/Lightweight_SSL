import torch
import logging

def print_info(model=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model:
        tot_params = sum(p.numel() for p in feature_extractor.parameters() if p.requires_grad)
        print(f"[INFO] {feature_extractor.name} loaded in memory.")
        print(f"[INFO] Feature size: {feature_extractor.feature_size}")
        print(f"[INFO] Feature extractor TOT trainable params: {tot_params}")
    if(torch.cuda.is_available() == False): 
        print("[WARNING] CUDA is not available.")
    else:
        print(f"[INFO] Found {torch.cuda.device_count()} GPU(s) available.")

    print(f"[INFO] Device type: {device}") 


def load_ckpt(model, checkpoint):
    # NOTE: the checkpoint must be loaded AFTER 
    # the model has been allocated into the device.
    if(checkpoint!=""):
        print("Loading checkpoint: " + str(checkpoint))
        model.load(checkpoint)
        print("Loading checkpoint: Done!")
    return model


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


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