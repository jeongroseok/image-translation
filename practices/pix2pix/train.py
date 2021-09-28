import pytorch_lightning as pl
from pl_examples import _DATASETS_PATH

from callbacks import Pix2PixVisualizer
from datamodules import FacadesDataModule
from models.pix2pix import Pix2Pix
from utils import *


def main(args=None):
    set_persistent_workers(FacadesDataModule)

    dm = FacadesDataModule(
        _DATASETS_PATH,
        num_workers=1,
        batch_size=1,
        shuffle=True,
        normalize=True
    )

    model = Pix2Pix(in_channels=3, out_channels=3)

    callbacks = [
        Pix2PixVisualizer(),
    ]
    trainer = pl.Trainer(
        progress_bar_refresh_rate=10,
        max_epochs=150,
        callbacks=callbacks,
        gpus=-1 if dm.num_workers > 0 else None,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
