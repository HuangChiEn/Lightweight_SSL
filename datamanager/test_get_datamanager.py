from datamanager.transformer import Transform_builder
from datamanager.data_module import DataModule

ds_name = "cifar100"
trfs_builder = Transform_builder(ds_name)
trfs = trfs_builder.prepare_n_crop_transform([2])

dm = DataModule(data_dir='/data/tch_ds', ds_name=ds_name)
dm.train_transform = trfs
dm.setup(stage='train')
for im, lab in dm.train_dataloader():
    print(f"len of label : {len(lab)}\n") # 32
    print(f"im shape : {im[0].shape}\n")  # (32), 3, 32, 32
    print(f"num of view {len(im)}")       # 2 view, im[0], im[1] indicate same img with diff view..
    break

