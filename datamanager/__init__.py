from datamanager.transformer import Transform_builder
from datamanager.data_module import DataModule

def get_datamanager(ds_name, aug_crop_lst, custom_ds_cfg=None):    
    if custom_ds_cfg:
        pass
    else:
        trfs_builder = Transform_builder(ds_name)
        dm = DataModule(data_dir='/data/tch_ds', ds_name=ds_name)
        dm.train_transform = trfs_builder.prepare_n_crop_transform(aug_crop_lst)
        return dm
