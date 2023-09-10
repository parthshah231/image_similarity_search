from pathlib import Path

import pytorch_lightning as pl

from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from configs import Config
from model import SiameseNetwork


def train_siamese(
    config: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> None:
    """The below method trains and stores a siamese network
    along with a specified config.

    Params:
    -------
    config: Config

    train_loader: DataLoader

    val_loader: DataLoader

    Returns:
    --------
    None"""
    model = SiameseNetwork(config=config)

    model_checkpoint = ModelCheckpoint(
        filename="{epoch}_{val_loss:1.3f}",
        monitor="val/triplet",
        save_last=True,
        save_top_k=1,
        mode="min",
        auto_insert_metric_name=True,
        save_weights_only=False,
        every_n_epochs=1,
    )

    early_stopping = EarlyStopping(
        monitor="val/triplet",
        check_finite=True,
        patience=30,
        verbose=False,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=config.epochs, callbacks=[model_checkpoint, early_stopping]
    )
    if trainer.log_dir is None:
        raise ValueError("No logging dir was found.")
    log_version_dir = Path(trainer.log_dir)

    if not log_version_dir.exists():
        log_version_dir.mkdir(parents=True, exist_ok=True)

    trainer.fit(model, train_loader, val_loader)
    print("Model trained.")

    config.save(log_version_dir)
    print(f"Configs saved at {log_version_dir}")


if __name__ == "__main__":
    pass
