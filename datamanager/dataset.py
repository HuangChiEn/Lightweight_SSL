import torchvision.datasets as dset
from typing import Tuple
from dataclasses import dataclass

## HACKME : still complete the dataset catelogs
## Note, the benchmark dataset in SSL
#  provided by How Well Do Self-Supervised Models Transfer? (2021/03)
@dataclass
class InfoSpec:
    num_classes: int
    img_shape: Tuple[int, int, int]
    online: bool
    split: bool   # split_type: str = ['train', 'split']

# Collect the dataset information from tfds catelog : https://www.tensorflow.org/datasets/catalog/overview?hl=zh-tw#all_datasets
DS_DICT = {
    #  small-scale ds : 
    'cifar10' : { 'data' : getattr(dset, 'CIFAR10'), 'info' : InfoSpec(10, (3, 32, 32), True, True) },
    'cifar100' : { 'data' : getattr(dset, 'CIFAR100'), 'info' : InfoSpec(100, (3, 32, 32), True, True) },
    'imagenet' : { 'data' : getattr(dset, 'ImageFolder'), 'info' : InfoSpec(1000, (3, 224, 224), False, False) },
    'stl10' : { 'data' : getattr(dset, 'STL10'), 'info' : InfoSpec(10, (3, 96, 96), True, False) },
    'slim' : { 'data' : getattr(dset, 'ImageFolder'), 'info' : InfoSpec(1000, (3, 64, 64), False, False) }
}

# Not Implemented
    #'LSUN' : getattr(dset, 'LSUN'), 
    #'LSUNClass' : getattr(dset, 'LSUNClass'),  
    #'Caltech256' : getattr(dset, 'Caltech256'),
    #'CocoCaptions' : getattr(dset, 'CocoCaptions'),
    #'CocoDetection' : getattr(dset, 'CocoDetection'),
    #'VOCDetection' : getattr(dset, 'VOCDetection'), 
    #'VOCSegmentation' : getattr(dset, 'VOCSegmentation'), 