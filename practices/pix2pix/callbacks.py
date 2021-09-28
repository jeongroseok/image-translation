from abc import abstractmethod

import torch
import torchvision.utils
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from torch import Tensor
from torch.utils.tensorboard.writer import SummaryWriter

from datasets import Pix2PixDataset


class ImageVisualizer(Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        writer: SummaryWriter = trainer.logger.experiment

        with torch.no_grad():
            pl_module.eval()
            img = self._forward(trainer, pl_module)
            pl_module.train()

        tag = self._create_tag(pl_module)

        writer.add_image(tag, img, global_step=trainer.current_epoch)

    @abstractmethod
    def _forward(self, trainer: Trainer, pl_module: LightningModule):
        pass

    @abstractmethod
    def _create_tag(self, pl_module: LightningModule):
        pass


class Pix2PixVisualizer(ImageVisualizer):
    from models.pix2pix import Pix2Pix

    def _forward(self, trainer: Trainer, pl_module: Pix2Pix):
        dataloader: Pix2PixDataset = trainer.train_dataloader

        outputs: Tensor = None
        for inputs, targets in dataloader:
            inputs = inputs.to(pl_module.device)
            targets = targets.to(pl_module.device)
            outputs = pl_module.forward(inputs)
            break

        images = torch.cat([inputs, outputs, targets], 0)
        img = torchvision.utils.make_grid(
            images, outputs.shape[0], normalize=True, value_range=(-1., 1.))
        return img

    def _create_tag(self, pl_module: LightningModule):
        return f'{pl_module.__class__.__name__}'
