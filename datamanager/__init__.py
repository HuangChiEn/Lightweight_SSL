from datamanager.transformer import Transform_builder
from datamanager.data_module import DataModule

def get_datamanager(ds_name, data_dir, aug_crop_lst):   
    trfs_builder = Transform_builder(ds_name)
    dm = DataModule(ds_name=ds_name, data_dir=data_dir)
    dm.train_transform = trfs_builder.prepare_n_crop_transform(aug_crop_lst)
    return dm
