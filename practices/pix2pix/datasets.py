import os
from abc import abstractmethod
from typing import Optional

import numpy as np
from PIL import Image
from torch import Tensor
from torchvision.datasets import VisionDataset


class Pix2PixDataset(VisionDataset):
    @abstractmethod
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        pass


class Facades(Pix2PixDataset):
    _repr_indent = 4

    def __init__(
        self,
        root: str,
        mode: str = 'train',
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
    ):
        super().__init__(root, transform=transform,
                         target_transform=target_transform)

        self.root = os.path.expanduser(root)
        self.mode = mode
        self.transform = transform

        if mode not in ['train', 'val', 'test']:
            raise ValueError('invalid mode')

        self.samples = self.__load_data()

    def __load_data(self):
        samples = []
        for root, _, fnames in os.walk(self.__folder, followlinks=True):
            for fname in fnames:
                path = os.path.join(root, fname)
                samples.append(path)
        return samples

    @property
    def __folder(self):
        return os.path.join(self.root, self.__class__.__name__, self.mode)

    def __getitem__(self, index: int):
        path = self.samples[index]
        sample = Image.open(path)
        sample = np.asarray(sample)

        width = int(sample.shape[1] / 2)

        input = Image.fromarray(sample[:, width:, :])
        target = Image.fromarray(sample[:, :width, :])

        if self.transform is not None:
            input = self.transform(input)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.samples)
