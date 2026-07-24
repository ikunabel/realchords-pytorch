"""Base Trainer class for training model via PyTorch Lightning."""

# References:
# https://github.com/Lightning-AI/pytorch-lightning/blob/master/examples/fabric/build_your_own_trainer/trainer.py
# https://lightning.ai/docs/pytorch/stable/common/trainer.html
# https://lightning.ai/docs/pytorch/stable/model/train_model_basic.html
# https://lightning.ai/docs/fabric/stable/guide/lightning_module.html

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from typing import Optional

from pathlib import Path
from lightning.fabric.strategies import FSDPStrategy
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import seed_everything
from lightning.pytorch.utilities import grad_norm

from realchords.utils.log_utils import midi_to_audio_image
from realchords.utils.lr_scheduler import LinearWarmupCosineDecay
from realchords.constants import MIDI_SYNTH_SR, LOG_WANDB_MIDI_IMAGE


class Trainer:
    """Base step-based Trainer class for training model via PyTorch Lightning.

    We give lightning module, dataloaders, and fabric arguments to the trainer.
    In the trainer, it will create ultimately the lightning trainer and train the model.
    We use Fabric explicitly to manage distributed training.
    """

    def __init__(
        self,
        args: dict,
        lit_module: L.LightningModule,
        save_dir: str,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        seed: int = 42,
        # Training args
        train_steps: int = 1000,
        val_interval: int = 1000,
        # You need to define sample_interval in module if you want to use it
        limit_val_batches: Optional[int] = None,
        overfit_batches: int = 0,  # set to 1 for debugging
        # Logging args
        wandb_project: str = "realchords",
        log_every_n_steps: int = 1,
        auto_export_wandb: bool = True,
        wandb_export_dir: str = "scripts/wandb/exports",
        # Checkpointing args
        checkpoint_interval: int = 0,  # set to 0 to disable
        checkpoint_metric: str = "val/loss",
        checkpoint_mode: str = "min",
        checkpoint_top_k: int = 2,
        early_stopping_patience: Optional[int] = None,
        early_stopping_min_delta: float = 0.0,
        # Fabric/distribution training args
        use_fabric: bool = False,
        accelerator: str = "auto",
        devices: str = "auto",
        num_nodes: int = 1,
        precision: str = None,
        strategy: str = "auto",
        # Gradient clipping args
        gradient_clip_val: float = 1.0,
    ):
        super().__init__()
        self.args = args  # the overall arguments
        self.save_dir = Path(save_dir)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # For GPUs with Tensor Cores
        torch.set_float32_matmul_precision("high")

        if use_fabric:
            # Setup distribution training
            if strategy == "fsdp":
                strategy = FSDPStrategy(state_dict_type="full")
            self.fabric = L.Fabric(
                accelerator=accelerator,
                num_nodes=num_nodes,
                precision=precision,
                devices=devices,
                strategy=strategy,
            )
            self.fabric.launch()
            self.lit_module = self.fabric.setup(lit_module)
            self.fabric.barrier()
            self.fabric.seed_everything(seed)
            self.distributed = True
            self.wandb_offline = not self.fabric.is_global_zero

        else:
            seed_everything(seed)
            self.lit_module = lit_module
            self.distributed = False
            self.wandb_offline = False

        # Checkpointing
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.save_dir,
            filename="{step}",  # save by step
            every_n_train_steps=checkpoint_interval,
            monitor=checkpoint_metric,
            mode=checkpoint_mode,
            save_top_k=checkpoint_top_k,
            enable_version_counter=False,  # overwrite the same file
        )

        # Log learning rate
        lr_monitor = LearningRateMonitor(logging_interval="step")

        callbacks = [checkpoint_callback, lr_monitor]
        if early_stopping_patience is not None and early_stopping_patience > 0:
            callbacks.append(
                EarlyStopping(
                    monitor=checkpoint_metric,
                    mode=checkpoint_mode,
                    patience=early_stopping_patience,
                    min_delta=early_stopping_min_delta,
                    check_on_train_epoch_end=False,
                )
            )

        # No wandb logging for processes other than rank 0
        logger = WandbLogger(
            name=self.save_dir.name,
            project=wandb_project,
            log_model=False,
            save_dir=self.save_dir,
            offline=self.wandb_offline,
            config=args,
        )

        # Capture the run path now (pre-training) rather than after `.fit()`,
        # since touching `logger.experiment` post-finish can implicitly start
        # a new run in some wandb versions. Offline runs (non-rank-0, or
        # WANDB_MODE=offline) aren't synced to the server, so there's nothing
        # for the W&B API to fetch -- skip those.
        self.auto_export_wandb = auto_export_wandb
        self.wandb_export_dir = wandb_export_dir
        self._wandb_run_path = None
        if auto_export_wandb and not self.wandb_offline:
            experiment = logger.experiment
            self._wandb_run_path = (
                f"{experiment.entity}/{experiment.project}/{experiment.id}"
            )

        self.trainer = L.Trainer(
            max_steps=train_steps,
            val_check_interval=val_interval,
            logger=logger,
            log_every_n_steps=log_every_n_steps,
            callbacks=callbacks,
            limit_val_batches=limit_val_batches,
            overfit_batches=overfit_batches,
            check_val_every_n_epoch=None,
            accelerator=accelerator,
            devices=devices,
            num_nodes=num_nodes,
            precision=precision,
            strategy=strategy,
            gradient_clip_val=gradient_clip_val,
        )

    def train(self):
        """The entry point for training the model."""
        self.trainer.fit(
            self.lit_module, self.train_dataloader, self.val_dataloader
        )
        if self._wandb_run_path is not None:
            self._export_wandb_run()

    def _export_wandb_run(self):
        """Dump this run's history/config/summary locally right after training
        finishes, so results are available for offline analysis without a
        manual export step (see scripts/wandb/export_run.py for the same logic
        as a standalone CLI, e.g. to backfill older runs).
        """
        try:
            import wandb

            from realchords.utils.wandb_export import export_run

            api = wandb.Api()
            run_dir = export_run(api, self._wandb_run_path, self.wandb_export_dir)
            print(f"Exported W&B run to {run_dir}")
        except Exception as e:
            print(
                f"Warning: failed to auto-export W&B run "
                f"{self._wandb_run_path}: {e}"
            )


class BaseLightningModel(L.LightningModule):
    """Base PyTorch Lightning model for ReaLchords.

    Create your own lightning module and pass it as `module` to the trainer.
    """

    def training_step(self, batch, batch_idx):
        """
        A single step during training. Calculates the loss for the batch and logs it.
        """
        pass

    def validation_step(self, batch, batch_idx):
        """
        A single step during validation. Calculates the loss for the batch and logs it.
        """
        pass

    def test_step(self, batch, batch_idx):
        """
        A single step during testing. Calculates the loss for the batch and logs it.
        """
        pass

    def configure_optimizers(self):
        """
        Configures and returns the optimizer(s).
        """
        pass

    def _configure_optimizer_with_schedule(self, optimizer, warmup_steps: int = 1000):
        """Wrap `optimizer` with linear-warmup + cosine decay over the full
        training run (`self.trainer.max_steps`), in Lightning's dict format.
        Without this, LR stays flat for the whole run, so the optimizer never
        slows down as training approaches overfitting -- it keeps taking
        full-sized steps into sharper minima that fit noise right up to the
        last step.
        """
        scheduler = LinearWarmupCosineDecay(
            optimizer,
            warmup_iters=warmup_steps,
            total_iters=self.trainer.max_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def _log_per_dataset_loss(
        self, per_sample_loss, dataset_names, key_prefix: str = "val/loss_by_dataset"
    ):
        """Log mean `per_sample_loss` grouped by `dataset_names` (parallel,
        both length batch_size). Uses the same per-step mean-of-means
        aggregation Lightning's default on_epoch reduction already applies to
        the aggregate `val/loss` metric, so the two are directly comparable.
        """
        per_dataset = {}
        for name in set(dataset_names):
            mask = torch.tensor(
                [n == name for n in dataset_names], device=per_sample_loss.device
            )
            per_dataset[f"{key_prefix}/{name}"] = per_sample_loss[mask].mean()
        self._log_dict(per_dataset)

    def _register_dataset_info(self, train_dataset):
        """Stash the train dataset's resolved per-dataset sampling info for later
        logging (see `on_train_start`). `train_dataset.dataset_info` already holds
        the actual weight used per dataset, whether it came from an explicit
        `weights` list, `alpha`-derived scaling, or the equal-weight default.
        `train_dataset.alpha` is the raw alpha passed in (None if not used).
        """
        self._dataset_info = getattr(train_dataset, "dataset_info", None)
        self._dataset_alpha = getattr(train_dataset, "alpha", None)

    def on_train_start(self):
        """
        Hook that is called at the very beginning of training (before any epochs).
        You can set up things like timers, experiment tracking, etc. here.
        """
        if getattr(self, "_dataset_info", None):
            resolved_weights = {
                f"dataset_weights/{info['name']}": info["weight"]
                for info in self._dataset_info
            }
            resolved_weights["dataset_weights/alpha"] = self._dataset_alpha
            self.logger.experiment.config.update(
                resolved_weights, allow_val_change=True
            )

    def on_train_end(self):
        """
        Hook that is called at the end of training (after all epochs).
        This is useful for cleanup, final logging, saving models, etc.
        """
        pass

    def get_dataloaders(self):
        return self.train_dataloader, self.val_dataloader

    def _log_scalar(self, name, value, **kwargs):
        self.log(
            name,
            value,
            prog_bar=True,
            logger=True,
            **kwargs,
        )

    def _log_dict(self, dict_, **kwargs):
        self.log_dict(dict_, prog_bar=True, logger=True, **kwargs)

    def log_midi(self, midi, suffix=""):
        audio, image = midi_to_audio_image(midi)
        payload = {}
        if image is not None:
            payload[f"image/{suffix}"] = wandb.Image(image)
        if audio is not None:
            payload[f"audio/{suffix}"] = wandb.Audio(
                audio,
                sample_rate=MIDI_SYNTH_SR,
            )
        if payload:
            self.logger.experiment.log(payload)

    def on_before_optimizer_step(self, optimizer):
        """
        Compute the 2-norm for each layer.
        If using mixed precision, the gradients are already unscaled here
        """
        norms = grad_norm(self.model, norm_type=2)
        self._log_dict({"grad_norm": norms["grad_2.0_norm_total"]})
        # print(f"global_step: {self.global_step}")
