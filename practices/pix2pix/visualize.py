import matplotlib.pyplot as plt
from pl_examples import _DATASETS_PATH
from torch import Tensor

from datamodules import FacadesDataModule
from models.pix2pix import Pix2Pix
from utils import *
import torchvision.utils


def main(args=None):
    dm = FacadesDataModule(
        _DATASETS_PATH,
        num_workers=0,
        batch_size=4,
        shuffle=True,
        normalize=True
    )

    dm.setup()

    model: Pix2Pix = Pix2Pix.load_from_checkpoint(
        fr'lightning_logs\version_4\checkpoints\epoch=149-step=59999.ckpt')

    for inputs, targets in dm.val_dataloader():
        outputs: Tensor = model.forward(inputs)
        img = torchvision.utils.make_grid(outputs, 2)
        img = img.permute(1, 2, 0).numpy()
        plt.imsave('_IMGS/outputs.png', img)
        break


if __name__ == "__main__":
    main()
