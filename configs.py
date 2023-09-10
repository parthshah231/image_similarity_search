import json
import pickle

from typing import Literal
from pathlib import Path


class Config:
    def __init__(
        self,
        backbone: Literal["resnext", "efficientnet", "mobilenet", "wide_resnet"],
        augment: bool,
        epochs: int,
        lr: float,
        wd: float,
    ) -> None:
        """Creates a config object that can be passed around across
        the project as a lightweight to carry necessary information.

        Params:
        -------
        backbone: Literal['resnext', 'efficientnet', 'mobilenet', 'wide_resnet']

        augment: bool

        epochs: int

        lr: float

        wd: float

        Returns:
        --------
        None"""
        self.backbone = backbone
        self.augment = augment
        self.epochs = epochs
        self.lr = lr
        self.wd = wd

    @classmethod
    def build(cls, config_path: Path):
        """Builds a config object given a config path."""
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        config = cls(**config_dict)
        return config

    def to_json(self, log_version_dir):
        """Converts the config dict and stores it in a json format
        (as a record incase we want to re-run the model or load an
        existing model)."""
        out_dir = log_version_dir / "configs.json"
        with open(out_dir, "w") as handle:
            config_dict = {**self.__dict__}
            new_dict = {}
            for key, dict_value in config_dict.items():
                value = dict_value
                new_dict[key] = value
            json.dump(new_dict, handle, default=str, indent=2)

    def to_pickle(self, log_version_dir):
        """Converts the config dict and stores it in a pickle format."""
        out_dir = log_version_dir / "configs.pickle"
        with open(out_dir, "wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def save(self, log_version_dir):
        """Save the config dict in json and pickle format."""
        self.to_json(log_version_dir=log_version_dir)
        self.to_pickle(log_version_dir=log_version_dir)

    def __repr__(self) -> str:
        """Readable representation of config dict."""
        fmt = (
            f"{self.__class__.__name__}(\n"
            f"\t         Backbone: {self.backbone}\n"
            f"\t          Augment: {self.augment}\n"
            f"\t    Learning Rate: {self.lr}\n"
            f"\t     Weight Decay: {self.wd}\n"
            ")"
        )
        return fmt
