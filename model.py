from typing import List, Tuple
from pathlib import Path

import pickle
import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import Tensor

from configs import Config
from loaders import TestDataset
from constants import ROOT


class TripletLoss(nn.Module):
    def __init__(self, alpha: float = 1.0):
        """Computes loss on the basis of embeddings from
        anchor, positive and negative image.

        Params:
        -------
        alpha: float = 1.0

        Returns:
        --------
        None"""
        super().__init__()
        self.margin = alpha

    def forward(
        self,
        anchor: Tensor,
        positive: Tensor,
        negative: Tensor,
    ) -> Tensor:
        """Uses Euclidean distance to measures distances between
        embeddings from anchor, positive and negative image and
        returns a mean from the losses tensor.

        Params:
        -------
        anchor: Tensor

        positive: Tensor

        negative: Tensor

        Returns:
        --------
        losses.mean(): Tensor"""
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()


class ImageEncoder(pl.LightningModule):
    def __init__(self, pretrained=True, backbone: str = "resnext") -> None:
        """Initialize a pre-trained encoder model for images, which could
        also be thought of as backbone of the structure

        Params:
        -------
        pretrained: bool

        backbone: str = resnext

        Returns:
        --------
        None"""
        super().__init__()

        # Initialize the backbone with a pre-trained model
        if backbone == "resnext":
            self.backbone = models.resnext50_32x4d(pretrained=pretrained)
        elif backbone == "efficientnet":
            self.backbone = models.efficientnet_v2_s(pretrained=pretrained)
        elif backbone == "mobilenet":
            self.backbone = models.mobilenet_v3_small(pretrained=pretrained)
        elif backbone == "wide_resnet":
            self.backbone = models.wide_resnet50_2(pretrained=pretrained)
        else:
            raise ValueError(f"{self.backbone} backbone is not implemented.")
        in_features = int(list(self.backbone.children())[-1].in_features)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # Freeze the early layers of the backbone
        for param in list(self.backbone.parameters())[:-27]:
            param.requires_grad = False

        # Define the additional encoding layers
        self.encoding_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            self.L2Normalization(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Returns embeddings of the image.

        Params:
        -------
        x: Tensor

        Returns:
        --------
        x: Tensor"""
        x = self.backbone(x)
        x = self.encoding_layers(x)
        return x

    class L2Normalization(nn.Module):
        def forward(self, x):
            return F.normalize(x, p=2, dim=1)


class SiameseNetwork(pl.LightningModule):
    def __init__(self, config: Config):
        """Buils a siamese net using PyTorch Lightning module and
        trains alongside triplet loss.

        Params:
        -------
        config: Config

        Returns:
        --------
        None"""
        super().__init__()
        self.config = config
        self.encoder = ImageEncoder()
        self.criterion = TripletLoss()
        self.test_dataset_cache = {}

    def forward(self, anchor, positive, negative):
        """Returns embeddings for anchor, positive and negative image.

        Params:
        -------
        anchor: Tensor

        postive: Tensor

        negative: Tensor"""
        encoded_anchor = self.encoder(anchor)
        encoded_positive = self.encoder(positive)
        encoded_negative = self.encoder(negative)
        return encoded_anchor, encoded_positive, encoded_negative

    def find_similar_images(
        self,
        input_image: Tensor,
        test_dataset: TestDataset,
        top_k: int = 5,
    ) -> Tuple[List[int], List[float]]:
        """Computes distances among each images in the dataset to find similarities and
        returns the images that lie closest to the input image.

        Params:
        -------
        input_image: Tensor

        test_dataset: TestDataset

        top_k: int = 5

        Returns:
        --------
        top_k_indices: List[int]

        top_k_similarities: List[float]"""
        self.eval()  # Set the network in evaluation mode

        # Creating a hash key for the test dataset
        test_dataset_key = hash(
            tuple(hash(img.cpu().numpy().tobytes()) for img, _ in test_dataset)
        )

        cache_file_path = Path(ROOT / f"cache_{test_dataset_key}.pkl")

        if cache_file_path.exists():
            with open(cache_file_path, "rb") as f:
                self.test_dataset_cache = pickle.load(f)
        else:
            # If the test dataset features are not in the cache, compute and store them
            if test_dataset_key not in self.test_dataset_cache:
                with torch.no_grad():
                    self.test_dataset_cache[test_dataset_key] = [
                        self.encoder(img.unsqueeze(0).to(self.device))
                        for img, _ in test_dataset
                    ]

            with open(cache_file_path, "wb") as f:
                pickle.dump(self.test_dataset_cache, f)

        with torch.no_grad():  # Disable gradient computation
            input_image = input_image.unsqueeze(0).to(self.device)
            input_image_feature = self.encoder(input_image)

            # Retrieve the cached test dataset features
            test_features = self.test_dataset_cache[test_dataset_key]

            # Compute the similarities between the input image and the test dataset
            similarities = [
                F.cosine_similarity(input_image_feature, test_feature, dim=-1).item()
                for test_feature in test_features
            ]

            # Find the top k most similar images
            top_k_indices = sorted(
                range(len(similarities)), key=lambda i: similarities[i], reverse=True
            )[1 : top_k + 1]
            top_k_similarities = [similarities[i] for i in top_k_indices]

        return top_k_indices, top_k_similarities

    def training_step(
        self,
        batch: Tuple[Tensor, Tensor, Tensor, int, int, int],
        batch_idx,
    ) -> Tensor:
        anchor, positive, negative, _, _, _ = batch
        output_anchor, output_positive, output_negative = self(
            anchor, positive, negative
        )
        loss = self.criterion(output_anchor, output_positive, output_negative)
        self.log("train/triplet", loss, prog_bar=True, on_step=True)
        return loss

    def validation_step(
        self,
        batch: Tuple[Tensor, Tensor, Tensor, int, int, int],
        batch_idx,
    ) -> Tensor:
        anchor, positive, negative, _, _, _ = batch
        output_anchor, output_positive, output_negative = self(
            anchor, positive, negative
        )
        loss = self.criterion(output_anchor, output_positive, output_negative)
        self.log("val/triplet", loss, prog_bar=True, on_step=True)
        return loss

    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.AdamW], List[CosineAnnealingLR]]:
        opt = torch.optim.AdamW(self.parameters(), lr=self.config.lr)
        sched = CosineAnnealingLR(optimizer=opt, T_max=self.config.epochs)
        return [opt], [sched]


if __name__ == "__main__":
    pass
