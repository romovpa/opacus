#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Runs MNIST training with differential privacy.
This example demonstrates how to use Opacus with PyTorch Lightning.

To start training:
$ python mnist_lightning.py fit

More information about setting training parameters:
$ python mnist_lightning.py fit --help

To see logs:
$ tensorboard --logdir=lightning_logs/
"""

import warnings
from typing import Optional, Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
from opacus.lightning import OpacusCallback
from pytorch_lightning.utilities.cli import LightningCLI
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


warnings.filterwarnings("ignore")


class LitSampleConvNetClassifier(pl.LightningModule):
    def __init__(
        self,
        lr: float = 0.1,
    ):
        """A simple conv-net for classifying MNIST with differential privacy

        Args:
            lr: Learning rate
        """
        super().__init__()

        # Hyper-parameters
        self.lr = lr

        # Parameters
        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

        # Privacy engine
        self.privacy_engine = None  # Created before training

        # Metrics
        self.test_accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        # x of shape [B, 1, 28, 28]
        x = F.relu(self.conv1(x))  # -> [B, 16, 14, 14]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 16, 13, 13]
        x = F.relu(self.conv2(x))  # -> [B, 32, 5, 5]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 32, 4, 4]
        x = x.view(-1, 32 * 4 * 4)  # -> [B, 512]
        x = F.relu(self.fc1(x))  # -> [B, 32]
        x = self.fc2(x)  # -> [B, 10]
        return x

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0)
        return optimizer

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = F.cross_entropy(output, target)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = F.cross_entropy(output, target)
        self.test_accuracy(output, target)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_accuracy", self.test_accuracy, on_step=False, on_epoch=True)
        return loss



class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        test_batch_size: int = 1000,
        data_dir: Optional[str] = "../mnist",
    ):
        """MNIST DataModule with DP-ready batch sampling

        Args:
            data_dir: A path where MNIST is stored
            test_batch_size: Size of batch for predicting on test
            sample_rate: Sample rate used for batch construction
            secure_rng: Use secure random number generator
        """
        super().__init__()
        self.data_root = data_dir
        self.dataloader_kwargs = {"num_workers": 1, "pin_memory": True}

        self.save_hyperparameters()

    @property
    def transform(self):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def prepare_data(self) -> None:
        datasets.MNIST(self.data_root, download=True)

    def train_dataloader(self):
        train_dataset = datasets.MNIST(
            self.data_root,
            train=True,
            download=False,
            transform=self.transform,
        )
        return DataLoader(
            train_dataset,
            **self.dataloader_kwargs,
        )

    def test_dataloader(self):
        return DataLoader(
            datasets.MNIST(
                self.data_root,
                train=False,
                download=False,
                transform=self.transform,
            ),
            batch_size=self.hparams.test_batch_size,
            shuffle=True,
            **self.dataloader_kwargs,
        )


def main():
    data = MNISTDataModule()
    model = LitSampleConvNetClassifier()

    privacy_callback = OpacusCallback()
    privacy_data = privacy_callback.wrap_datamodule(data)

    trainer = pl.Trainer(
        max_epochs=2,
        enable_model_summary=False,
        callbacks=[privacy_callback],
    )
    trainer.fit(model, privacy_data)

    trainer.test(model, data)
    trainer.test(model, privacy_data)  # both work



def cli_main():
    cli = LightningCLI(
        LitSampleConvNetClassifier,
        MNISTDataModule,
        save_config_overwrite=True,
        trainer_defaults={
            "max_epochs": 10,
            "enable_model_summary": False,
        },
        description="Training MNIST classifier with Opacus and PyTorch Lightning",
    )
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)


if __name__ == "__main__":
    #cli_main()
    main()
