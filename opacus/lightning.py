from typing import Optional, Any

import pytorch_lightning as pl
import torch
from opacus import PrivacyEngine
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from torch.utils.data import DataLoader


class DPLightningDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datamodule: pl.LightningDataModule,
        sample_rate: float = 0.001,
        generator: Optional[torch.Generator] = None,
    ):
        super().__init__()
        self.datamodule = datamodule
        self.sample_rate = sample_rate
        self.generator = generator

    def wrap_dataloader(self, dataloader: DataLoader) -> DataLoader:
        batch_sampler = UniformWithReplacementSampler(
            num_samples=len(dataloader.dataset),
            sample_rate=self.sample_rate,
            generator=self.generator,
        )
        return DataLoader(
            # changed by the wrapper
            generator=self.generator,
            batch_sampler=batch_sampler,
            # inherited from the object
            dataset=dataloader.dataset,
            num_workers=dataloader.num_workers,
            collate_fn=dataloader.collate_fn, # wrap_collate_with_empty(collate_fn, sample_empty_shapes),
            pin_memory=dataloader.pin_memory,
            drop_last=dataloader.drop_last,
            timeout=dataloader.timeout,
            worker_init_fn=dataloader.worker_init_fn,
            multiprocessing_context=dataloader.multiprocessing_context,
            prefetch_factor=dataloader.prefetch_factor,
            persistent_workers=dataloader.persistent_workers,
        )

    def prepare_data(self) -> None:
        self.datamodule.prepare_data()

    def setup(self, stage: Optional[str] = None) -> None:
        self.datamodule.setup(stage)

    def train_dataloader(self):
        dataloader = self.datamodule.train_dataloader()
        return self.wrap_dataloader(dataloader)

    def val_dataloader(self):
        return self.datamodule.val_dataloader()

    def test_dataloader(self):
        return self.datamodule.test_dataloader()

    def predict_dataloader(self):
        return self.datamodule.predict_dataloader()

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        return self.datamodule.transfer_batch_to_device(batch, device, dataloader_idx)

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        return self.datamodule.on_before_batch_transfer(batch, dataloader_idx)

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        return self.datamodule.on_after_batch_transfer(batch, dataloader_idx)


class LightningPrivacyEngine(pl.Callback):

    def __init__(
        self,
        delta: float = 1e-5,
        sample_rate: float = 0.001,
        sigma: float = 1.0,
        max_per_sample_grad_norm: float = 1.0,
        secure_rng: bool = False,

    ):
        """Callback enabling differential privacy learning

        Args:
            delta: Target delta for which (eps, delta)-DP is computed
            sample_rate: Sample rate used for batch construction
            sigma: Noise multiplier
            max_per_sample_grad_norm: Clip per-sample gradients to this norm
            secure_rng: Use secure random number generator
        """
        self.delta = delta
        self.sample_rate = sample_rate
        self.sigma = sigma
        self.max_per_sample_grad_norm = max_per_sample_grad_norm
        self.secure_rng = secure_rng

        if secure_rng:
            try:
                import torchcsprng as prng
            except ImportError as e:
                msg = (
                    "To use secure RNG, you must install the torchcsprng package! "
                    "Check out the instructions here: https://github.com/pytorch/csprng#installation"
                )
                raise ImportError(msg) from e
            self.generator = prng.create_random_device_generator("/dev/urandom")
        else:
            self.generator = None

        self.original_dataloader = None

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        pl_module.privacy_engine = PrivacyEngine(
            pl_module,
            sample_rate=self.sample_rate,
            alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
            noise_multiplier=self.sigma,
            max_grad_norm=self.max_per_sample_grad_norm,
            secure_rng=self.secure_rng,
        )

        optimizer = pl_module.optimizers()
        pl_module.privacy_engine.attach(optimizer.optimizer)

        # TODO: check data loader is DP-compatible

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        epsilon, best_alpha = pl_module.privacy_engine.get_privacy_spent(self.delta)
        # Privacy spent: (epsilon, delta) for alpha
        pl_module.log("epsilon", epsilon, on_epoch=True, prog_bar=True)
        pl_module.log("alpha", best_alpha, on_epoch=True, prog_bar=True)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        pl_module.privacy_engine.detach()
        trainer.train_dataloader.loaders = self.original_dataloader
        self.original_dataloader = None

    def wrap_datamodule(self, datamodule: pl.LightningDataModule) -> DPLightningDataModule:
        return DPLightningDataModule(
            datamodule,
            sample_rate=self.sample_rate,
            generator=self.generator,
        )

