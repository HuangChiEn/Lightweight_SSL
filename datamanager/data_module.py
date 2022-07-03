# software-engerneering pkgs
from typing import Any, Callable, List, Optional, Sequence, Type, Union

# torch eco-system
from pytorch_lightning import LightningDataModule

#  dataset, trfs
from datamanager.dataset import DS_DICT
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, random_split


class DataModule(LightningDataModule):
    # /data is the default path setup by docker env, and bind to /data1 datasets
    def __init__(self, ds_name, data_dir):
        super().__init__()
        if ds_name not in DS_DICT:
            raise KeyError(f"The given dataset '{ds_name}' have not been supported, check the supported data catelog : {DS_DICT.keys()}\n")

        self.ds_name = ds_name
        self.data_dir = data_dir
        if isinstance(data_dir, dict):
            self.split_dir = True 
            for k in data_dir.keys():
                if not k in ['train', 'test']:
                    raise KeyError(f"The {k} is invalid, valid phase : [train, test]")
        else:
            self.split_dir = False

        self._train_trfs = transforms.ToTensor()
        self._test_trfs = transforms.ToTensor()

    ## Declared properties
    @property
    def dataset_info(self) -> int:
        return DS_DICT[self.ds_name]['info']

    @property
    def train_transform(self):
        return self._train_trfs

    @train_transform.setter
    def train_transform(self, new_trfs):
        self._train_trfs = new_trfs

    @property
    def test_transform(self):
        return self._test_trfs

    @test_transform.setter
    def test_transform(self, new_trfs):
        self._test_trfs = new_trfs

    def prepare_data(self):
        ds = DS_DICT[self.ds_name]

        # download dataset
        if ds['info'].online:
            if ds['info'].split:
                ds['data'](self.data_dir, train=True, download=True)
                ds['data'](self.data_dir, train=False, download=True)
            else:
                ds['data'](self.data_dir, download=True)
        else: # do it by yourself..
            print(f"warning.. The dataset {self.ds_name} is not avalible public")
            print(f"The given data_dir {self.data_dir} should be the parent folder of image subfolder.. plz follow the ImagenFolder instruction..")
            ds['data'](self.data_dir)


    def setup(self, stage = 'train', valid_split = None):
        self.prepare_data()

        ds = DS_DICT[self.ds_name]
        data_dir = self.data_dir[stage] if self.split_dir else self.data_dir
        # Assign train/val datasets for use in dataloaders
        if stage == "train":
            full_tra_ds = ds['data'](data_dir, train=True, transform=self._train_trfs) if ds['info'].split \
                            else ds['data'](data_dir, transform=self._train_trfs)
            if valid_split:
                assert valid_split[0] + valid_split[1] == 1.0
                tra_ratio = valid_split[0]
        
                train_set_size = int(len(full_tra_ds)*tra_ratio)
                valid_set_size = len(full_tra_ds) - train_set_size
                self.train_dset, self.valid_dset = random_split(full_tra_ds, [train_set_size, valid_set_size])
            else:
                self.train_dset = full_tra_ds

        # Assign test dataset for use in dataloader(s)
        elif stage == "test":
            self.test_dset = ds['data'](data_dir, train=False, transform=self._test_trfs) if ds['info'].split \
                                else ds['data'](data_dir, transform=self._test_trfs)

        else:
            raise ValueError(f"Error stage : {stage}, it should be either 'train' or 'test'.")


    # overwrite base class methods
    def train_dataloader(self, batch_size=32, shuffle=True, num_workers=2, pin_memory=True):
        return DataLoader(self.train_dset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)

    def val_dataloader(self, batch_size=32, shuffle=True, num_workers=2, pin_memory=True):
        return DataLoader(self.valid_dset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

    def test_dataloader(self, batch_size=32, shuffle=True, num_workers=2, pin_memory=True):
        return DataLoader(self.test_dset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)


if __name__ == "__main__":
    # unit test
    dm = DataModule(data_dir='/data/tch_ds', ds_name='stl10')
    dm.setup(stage='train')
    for im, lab in dm.train_dataloader():
        print(f"label : {lab[0]}\n")
        print(f"im shape : {im.shape}\n")
        break