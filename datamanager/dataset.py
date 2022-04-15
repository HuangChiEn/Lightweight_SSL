import torchvision.datasets as dset
from typing import Tuple
from dataclasses import dataclass

## HACKME : still complete the dataset catelogs
## Note, the benchmark dataset in SSL
#  provided by How Well Do Self-Supervised Models Transfer? (2021/03)
DS_DICT = {
    #  small-scale ds : 
    #   1. FGVC Aircraft (x),  2. Standford Cars (x),  3. Caltech101,  4. CIFAR10/100,  5. DTD(x)
    #   6. Flowers(x),  7. Food(x),  8. Pets(x),  9. SUN397(x),  10. VOC2007
    
    'Cifar10' : getattr(dset, 'CIFAR10'),
    'Cifar100' : getattr(dset, 'CIFAR100'),
    'Caltech101' : getattr(dset, 'Caltech101'),
    'Caltech256' : getattr(dset, 'Caltech256'),
    'CelebA' : getattr(dset, 'CelebA'),
    'Cityscapes' : getattr(dset, 'Cityscapes'),
    'CocoCaptions' : getattr(dset, 'CocoCaptions'),
    'CocoDetection' : getattr(dset, 'CocoDetection'),

    # E :
    'EMNIST' : getattr(dset, 'EMNIST'),
    
    # F :
    'FakeData' : getattr(dset, 'FakeData'),
    'FashionMNIST' : getattr(dset, 'FashionMNIST'),
    'Flickr30k' : getattr(dset, 'Flickr30k'), 
    'Flickr8k' : getattr(dset, 'Flickr8k'), 

    # H :
    'HMDB51' : getattr(dset, 'HMDB51'), 
    
    # I :
    'ImageNet' : getattr(dset, 'ImageNet'), 

    # K :
    'KMNIST' : getattr(dset, 'KMNIST'), 
    'Kinetics400' : getattr(dset, 'Kinetics400'), 
    'Kitti' : getattr(dset, 'Kitti'), 
    
    # L :
    'LSUN' : getattr(dset, 'LSUN'), 
    'LSUNClass' : getattr(dset, 'LSUNClass'),  
    
    # M :
    'MNIST' : getattr(dset, 'MNIST'), 
    
    # O :
    'Omniglot' : getattr(dset, 'Omniglot'), 
    
    # P :
    'PhotoTour' : getattr(dset, 'PhotoTour'), 
    'Places365' : getattr(dset, 'Places365'), 
    
    # Q :
    'QMNIST' : getattr(dset, 'QMNIST'), 
    
    # S :
    'SBDataset' : getattr(dset, 'SBDataset'), 
    'SBU' : getattr(dset, 'SBU'), 
    'SEMEION' : getattr(dset, 'SEMEION'), 
    'STL10' : getattr(dset, 'STL10'), 
    'SVHN' : getattr(dset, 'SVHN'), 
    
    # U :
    'UCF101' : getattr(dset, 'UCF101'), 
    'USPS' : getattr(dset, 'USPS'), 
    
    'VOCDetection' : getattr(dset, 'VOCDetection'), 
    'VOCSegmentation' : getattr(dset, 'VOCSegmentation'), 
    
    'WIDERFace' : getattr(dset, 'WIDERFace'), 

}


@dataclass
class InfoSpec:
    num_classes: int
    total_img: int
    img_shape: Tuple[int, int, int]
    
# Collect the dataset information from tfds catelog : https://www.tensorflow.org/datasets/catalog/overview?hl=zh-tw#all_datasets
DS_INFO = {
    'CIFAR10' : InfoSpec(10, 60000, (3, 32, 32)),
    'CIFAR100' : InfoSpec(100, 60000, (3, 32, 32)),

}

