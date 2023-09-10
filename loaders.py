from typing import Literal

import random

import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset

from configs import Config
from constants import DATA_PATH, LABEL_DICT


class SnacksDataset(Dataset):
    def __init__(
        self,
        config: Config,
        split: Literal["train", "validation"] = "train",
    ) -> None:
        """Build a PyTorch dataset (following a Huggingface DATAPATH -
        stored in `constants.py`) using a specified config and a desired split.

        Params:
        -------
        config: Config

        split: Literal['train', 'validation'] = 'train'

        Returns:
        --------
        None"""
        self.config = config
        self.dataset = load_dataset(DATA_PATH, split=split)

        if not self.config.augment:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.RandomRotation(degrees=15),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                ]
            )

        self.labels = [item["label"] for item in self.dataset]
        self.label_to_indices = {label: [] for label in set(self.labels)}

        for idx, label in enumerate(self.labels):
            self.label_to_indices[label].append(idx)

    def __len__(self):
        """Returns the length of dataset"""
        return len(self.labels)

    def __getitem__(self, idx: int):
        """Returns an anchor image, positive image (one with the same label as anchor image)
        and a negative image (one with a different index from anchor) along with
        their respective labels.

        Params:
        -------
        idx: int

        Returns:
        --------
        anchor_img: Tensor

        positive_img: Tensor

        negative_img: Tensor

        anchor_label: int

        positive_label: int

        negative_label: int"""
        anchor_img = self.transform(self.dataset[idx]["image"])
        anchor_label = self.dataset[idx]["label"]

        positive_idx = idx
        while positive_idx == idx:
            positive_idx = random.choice(self.label_to_indices[anchor_label])

        negative_label = random.choice(list(set(self.labels) - {anchor_label}))
        negative_idx = random.choice(self.label_to_indices[negative_label])

        positive_img = self.transform(self.dataset[positive_idx]["image"])
        negative_img = self.transform(self.dataset[negative_idx]["image"])

        positive_label = self.dataset[positive_idx]["label"]
        negative_label = self.dataset[negative_idx]["label"]

        return (
            anchor_img,
            positive_img,
            negative_img,
            anchor_label,
            positive_label,
            negative_label,
        )


class TestDataset(Dataset):
    def __init__(self):
        """Build a PyTorch test dataset following a test split."""
        self.dataset = load_dataset(DATA_PATH, split="test")

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        """Returns the length of dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """Returns an image and corresponding label.
        Params:
        -------
        idx: int

        Returns:
        --------
        img: Tensor

        label: int"""
        img = self.transform(self.dataset[idx]["image"])
        label = self.dataset[idx]["label"]
        return img, label


if __name__ == "__main__":
    # dummy test (works as expected)
    config = Config(
        backbone="resnext",
        augment=False,
        epochs=1,
        lr=1e-4,
        wd=1e-2,
    )
    ds = SnacksDataset(config=config, split="train")

    for i in range(10):
        a_img, p_img, n_img, a_label, p_label, n_label = ds[i]
        fig, ax = plt.subplots(nrows=1, ncols=3)
        ax[0].imshow(a_img.permute(1, 2, 0))
        ax[1].imshow(p_img.permute(1, 2, 0))
        ax[2].imshow(n_img.permute(1, 2, 0))
        ax[0].set_title(LABEL_DICT[str(a_label)])
        ax[1].set_title(LABEL_DICT[str(p_label)])
        ax[2].set_title(LABEL_DICT[str(n_label)])
        plt.show()
