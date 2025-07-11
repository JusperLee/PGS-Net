import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections.abc import MutableMapping
from speechbrain.processing.speech_augmentation import SpeedPerturb

def flatten_dict(d, parent_key="", sep="_"):
    """Flattens a dictionary into a single-level dictionary while preserving
    parent keys. Taken from
    `SO <https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys>`_

    Args:
        d (MutableMapping): Dictionary to be flattened.
        parent_key (str): String to use as a prefix to all subsequent keys.
        sep (str): String to use as a separator between two key levels.

    Returns:
        dict: Single-level dictionary, flattened.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class AudioAugLightningModule(pl.LightningModule):
    def __init__(
        self,
        audio_model=None,
        video_model=None,
        optimizer=None,
        loss_func=None,
        train_loader=None,
        val_loader=None,
        test_loader=None,
        scheduler=None,
        config=None,
    ):
        super().__init__()
        self.audio_model = audio_model
        self.video_model = video_model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scheduler = scheduler
        self.config = {} if config is None else config
        # Speed Aug
        self.speedperturb = SpeedPerturb(
            self.config["datamodule"]["data_config"]["sample_rate"],
            speeds=[95, 100, 105],
            perturb_prob=1.0
        )
        # Save lightning"s AttributeDict under self.hparams
        self.default_monitor = "val_loss/dataloader_idx_0"
        self.save_hyperparameters(self.config_to_hparams(self.config))
        # self.print(self.audio_model)
        

    def forward(self, wav, mouth=None):
        """Applies forward pass of the model.

        Returns:
            :class:`torch.Tensor`
        """
        return self.audio_model(wav)

    def training_step(self, batch, batch_nb):
        mixtures, targets, noise, _ = batch
        
        new_targets = []
        min_len = -1
        if self.config["training"]["SpeedAug"] == True:
            with torch.no_grad():
                for i in range(targets.shape[1]):
                    new_target = self.speedperturb(targets[:, i, :])
                    new_targets.append(new_target)
                    if i == 0:
                        min_len = new_target.shape[-1]
                    else:
                        if new_target.shape[-1] < min_len:
                            min_len = new_target.shape[-1]

                targets = torch.zeros(
                            targets.shape[0],
                            targets.shape[1],
                            min_len,
                            device=targets.device,
                            dtype=torch.float,
                        )
                for i, new_target in enumerate(new_targets):
                    targets[:, i, :] = new_targets[i][:, 0:min_len]
                    
                mixtures = targets.sum(1)
                
                len_noise = noise.shape[-1]
                len_mix = mixtures.shape[-1]
                min_len = min(len_noise, len_mix)

                # add the noise
                mixtures = mixtures[:, :min_len] + noise[:, :min_len]

                # fix the length of targets also
                targets = targets[:, :, :min_len]
            
        est_sources = self(mixtures)
        loss = self.loss_func["train"](est_sources, targets)

        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )

        return {"loss": loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        train_loss = torch.mean(self.all_gather(avg_loss))

    def validation_step(self, batch, batch_nb, dataloader_idx):
        # cal val loss
        if dataloader_idx == 0:
            mixtures, targets, noise, _ = batch
            est_sources = self(mixtures)
            loss = self.loss_func["val"](est_sources, targets)
            self.log(
                "val_loss",
                loss,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                logger=True,
            )
            return {"val_loss": loss}

        # cal test loss
        if (self.trainer.current_epoch) % 10 == 0 and dataloader_idx == 1:
            mixtures, targets, noise, _ = batch
            est_sources = self(mixtures)
            tloss = self.loss_func["val"](est_sources, targets)
            self.log(
                "test_loss",
                tloss,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                logger=True,
            )
            return {"test_loss": tloss}

    def validation_epoch_end(self, outputs):
        # val
        avg_loss = torch.stack([x["val_loss"] for x in outputs[0]]).mean()
        val_loss = torch.mean(self.all_gather(avg_loss))
        self.log(
            "lr",
            self.optimizer.param_groups[0]["lr"],
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.logger.experiment.add_scalar(
            "learning_rate", self.optimizer.param_groups[0]["lr"], self.current_epoch
        )
        self.logger.experiment.add_scalar(
            "val_pit_sisnr", -val_loss, self.current_epoch
        )

        # test
        if (self.trainer.current_epoch) % 10 == 0:
            avg_loss = torch.stack([x["test_loss"] for x in outputs[1]]).mean()
            test_loss = torch.mean(self.all_gather(avg_loss))
            self.logger.experiment.add_scalar(
                "test_pit_sisnr", -test_loss, self.current_epoch
            )

    def configure_optimizers(self):
        """Initialize optimizers, batch-wise and epoch-wise schedulers."""
        if self.scheduler is None:
            return self.optimizer

        if not isinstance(self.scheduler, (list, tuple)):
            self.scheduler = [self.scheduler]  # support multiple schedulers

        epoch_schedulers = []
        for sched in self.scheduler:
            if not isinstance(sched, dict):
                if isinstance(sched, ReduceLROnPlateau):
                    sched = {"scheduler": sched, "monitor": self.default_monitor}
                epoch_schedulers.append(sched)
            else:
                sched.setdefault("monitor", self.default_monitor)
                sched.setdefault("frequency", 1)
                # Backward compat
                if sched["interval"] == "batch":
                    sched["interval"] = "step"
                assert sched["interval"] in [
                    "epoch",
                    "step",
                ], "Scheduler interval should be either step or epoch"
                epoch_schedulers.append(sched)
        return [self.optimizer], epoch_schedulers

    def train_dataloader(self):
        """Training dataloader"""
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        return [self.val_loader, self.test_loader]

    def on_save_checkpoint(self, checkpoint):
        """Overwrite if you want to save more things in the checkpoint."""
        checkpoint["training_config"] = self.config
        return checkpoint

    @staticmethod
    def config_to_hparams(dic):
        """Sanitizes the config dict to be handled correctly by torch
        SummaryWriter. It flatten the config dict, converts ``None`` to
        ``"None"`` and any list and tuple into torch.Tensors.

        Args:
            dic (dict): Dictionary to be transformed.

        Returns:
            dict: Transformed dictionary.
        """
        dic = flatten_dict(dic)
        for k, v in dic.items():
            if v is None:
                dic[k] = str(v)
            elif isinstance(v, (list, tuple)):
                dic[k] = torch.tensor(v)
        return dic
