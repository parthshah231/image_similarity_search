from typing import Tuple

import argparse
from pathlib import Path

import pytorch_lightning as pl

from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from configs import Config
from loaders import SnacksDataset, TestDataset
from model import SiameseNetwork


def get_loaders(config: Config) -> Tuple[DataLoader, DataLoader, TestDataset]:
    train_dataset = SnacksDataset(config=config, split="train")
    val_dataset = SnacksDataset(config=config, split="validation")
    test_dataset = TestDataset()

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    return train_loader, val_loader, test_dataset


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
    parser = argparse.ArgumentParser(description="Train a Siamese network")
    parser.add_argument(
        "--backbone",
        type=str,
        required=True,
        choices=["resnext", "efficientnet", "mobilenet", "wide_resnet"],
        help="Image Encoder for the Siamese model",
    )
    parser.add_argument(
        "--augment",
        type="store_true",
        default=False,
        help="Whether to augment the data",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--wd", type=float, default=1e-2, help="Weight decay")

    args = parser.parse_args()
    config = Config(**vars(args))
    train_loader, val_loader, test_dataset = get_loaders(config=config)
    train_siamese(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
    )
