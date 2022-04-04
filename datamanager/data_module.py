# software-engerneering pkgs
from typing import Any, Callable, List, Optional, Sequence, Type, Union

# torch eco-system
from pytorch_lightning import LightningDataModule

#  dataset, trfs
from . import DS_DICT, DS_INFO
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split


class DataModule(LightningDataModule):
    
    def __init__(self, data_dir: str = "./tch_ds", ds_name:str = ''):
        super().__init__()
        self.ds_name = ds_name
        self.data_dir = data_dir
        self.transform = None

    ## Declared properties
    @property
    def dataset_info(self) -> int:
        if self.ds_name not in DS_INFO:
            raise ValueError(f"The given dataset '{self.ds_name}' have not been recorded")
        return DS_INFO[self.ds_name]

    @property
    def normalize_transform(self):
        return self._norm_trfs

    @property
    def train_transform(self):
        return self._train_trfs

    @train_transform.setter
    def train_transform(self, new_trfs):
        self._train_trfs = new_trfs

    @property
    def valid_transform(self):
        return self._valid_trfs

    @valid_transform.setter
    def valid_transform(self, new_trfs):
        self._valid_trfs = new_trfs

    def prepare_data(self):
        # download training set
        DS_DICT[self.ds_name](self.data_dir, train=True, download=True)
        # download testing set
        DS_DICT[self.ds_name](self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None, valid_split: List[int, int] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "train" or stage is None:
            full_tra_ds =  DS_DICT[self.ds_name](self.data_dir, train=True, transform=self.transform)
            if valid_split:
                self.train_dset, self.valid_dset = random_split(full_tra_ds, valid_split)
            else:
                self.train_dset = full_tra_ds

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dset =  DS_DICT[self.ds_name](self.data_dir, train=False, transform=self.transform)

        if stage == "predict" or stage is None:
            self.pred_dset =  DS_DICT[self.ds_name](self.data_dir, train=False, transform=self.transform)

    # overwrite base class methods
    def train_dataloader(self, batch_size=32, shuffle=True, num_workers=2):
        return DataLoader(self.train_dset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def val_dataloader(self, batch_size=32, shuffle=True, num_workers=2):
        return DataLoader(self.valid_dset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def test_dataloader(self, batch_size=32, shuffle=True, num_workers=2):
        return DataLoader(self.test_dset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def predict_dataloader(self, batch_size=32, shuffle=True, num_workers=2):
        return DataLoader(self.pred_dset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
