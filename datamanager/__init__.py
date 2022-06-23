from datamanager.transformer import Transform_builder
from datamanager.data_module import DataModule

def get_pretrain_datamanager(ds_name, data_dir, aug_crop_lst):
    dm = DataModule(ds_name=ds_name, data_dir=data_dir)
    trfs_builder = Transform_builder(ds_name)
    dm.train_transform = trfs_builder.prepare_n_crop_transform(aug_crop_lst)
    return dm


def get_finetune_datamanager(ds_name, data_dir, aug_crop_lst, only_tune=False):
    dm = DataModule(ds_name=ds_name, data_dir=data_dir)
    dm.train_transform = Transform_builder(ds_name).prepare_n_crop_transform(aug_crop_lst)
    if not only_tune:
        dm.test_transform = Transform_builder(ds_name, train=False).prepare_n_crop_transform(aug_crop_lst)
    
    return dm