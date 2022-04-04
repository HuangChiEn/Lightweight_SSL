from transformer import prepare_trfs
from data_module import DataModule
from dataset import DS_DICT

def get_datamanager(dataset, aug_crop_lst):
    if not self.ds_name in DS_DICT:
        raise ValueError("The given dataset is not supported by torchvision.dataset")
        
    trfs = prepare_trfs(aug_crop_lst)
    return DataModule(dataset, trfs)


def get_custom_datamanager(dataset, aug_crop_lst, custom_ds_cfg):
    raise NotImplementedError('Not Implemented')
