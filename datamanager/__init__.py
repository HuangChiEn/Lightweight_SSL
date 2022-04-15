from datamanager.transformer import Transform_builder
from datamanager.data_module import DataModule
from datamanager.dataset import DS_DICT

def get_datamanager(ds_name, aug_crop_lst, custom_ds_cfg=None):    
    if custom_ds_cfg:
        pass
    else:
        trfs_builder = Transform_builder(ds_name)
        trfs = trfs_builder.prepare_n_crop_transform(aug_crop_lst)
    return DataModule(ds_name=ds_name)