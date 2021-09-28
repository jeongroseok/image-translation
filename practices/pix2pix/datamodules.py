import os
from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as transform_lib

from datasets import Facades


class FacadesDataModule(LightningDataModule):
    name = "facades"
    dims = (3, 256, 256)

    def __init__(
        self,
        data_dir: Optional[str] = None,
        num_workers: int = 16,
        normalize: bool = False,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = False,
        pin_memory: bool = False,
        drop_last: bool = False,
        *args: any,
        **kwargs: any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.num_workers = num_workers
        self.normalize = normalize
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    def default_transforms(self) -> callable:
        if self.normalize:
            transforms = transform_lib.Compose([
                transform_lib.ToTensor(), transform_lib.Normalize(mean=(0.5, ), std=(0.5, ))
            ])
        else:
            transforms = transform_lib.Compose([transform_lib.ToTensor()])

        return transforms

    def prepare_data(self, *args: any, **kwargs: any):
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            train_transforms = self.default_transforms(
            ) if self.train_transforms is None else self.train_transforms
            val_transforms = self.default_transforms(
            ) if self.val_transforms is None else self.val_transforms

            self.dataset_train = Facades(
                self.data_dir, mode='train', transform=train_transforms, target_transform=train_transforms)
            self.dataset_val = Facades(
                self.data_dir, mode='val', transform=val_transforms, target_transform=val_transforms)

        if stage == "test" or stage is None:
            test_transforms = self.default_transforms(
            ) if self.test_transforms is None else self.test_transforms
            self.dataset_test = Facades(
                self.data_dir, mode='test', transform=test_transforms, target_transform=test_transforms)

    def train_dataloader(self):
        return self._data_loader(self.dataset_train, shuffle=self.shuffle)

    def val_dataloader(self):
        return self._data_loader(self.dataset_val)

    def test_dataloader(self):
        return self._data_loader(self.dataset_test)

    def _data_loader(self, dataset: Dataset, shuffle: bool = False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
