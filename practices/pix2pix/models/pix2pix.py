import pytorch_lightning as pl

from .components import *


class Pix2Pix(pl.LightningModule):
    class __HPARAMS:
        lr: float
        lambda_recon: float
    hparams: __HPARAMS

    def __init__(
        self,
        in_channels,
        out_channels,
        lr=2e-4,
        lambda_recon=100,
        *args: any,
        **kwargs: any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.criterion_recon = torch.nn.L1Loss()
        self.criterion_adv = torch.nn.BCELoss()

        self.generator = Generator(in_channels, out_channels)
        self.discriminator = Discriminator(in_channels)

    def forward(self, c):
        return self.generator.forward(c)

    def _gen_step(self, x, c):
        x_hat = self.generator.forward(c)
        d = self.discriminator.forward(x_hat, c)

        loss_adv = self.criterion_adv(d, torch.ones_like(d))
        loss_recon = self.criterion_recon(x_hat, x)

        self.log(f"{self.__class__.__name__}/gen/adv", loss_adv)
        self.log(f"{self.__class__.__name__}/gen/recon", loss_recon)

        return loss_adv + (loss_recon * self.hparams.lambda_recon)

    def _disc_step(self, x, c):
        x_hat = self.generator.forward(c)
        d_real = self.discriminator.forward(x, c)
        d_fake = self.discriminator.forward(x_hat, c)
        loss_adv_real = self.criterion_adv(d_real, torch.ones_like(d_real))
        loss_adv_fake = self.criterion_adv(d_fake, torch.zeros_like(d_fake))

        loss_adv = (loss_adv_fake + loss_adv_real) / 2

        self.log(f"{self.__class__.__name__}/disc/adv", loss_adv)
        return loss_adv

    def configure_optimizers(self):
        opt_gen = torch.optim.Adam(
            self.generator.parameters(), lr=self.hparams.lr)
        opt_disc = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.hparams.lr)
        return [opt_disc, opt_gen]

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs, targets = batch

        if optimizer_idx == 0:
            return self._disc_step(targets, inputs)
        else:
            return self._gen_step(targets, inputs)
