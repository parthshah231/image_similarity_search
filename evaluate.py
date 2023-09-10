from typing import List, Tuple
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor

from configs import Config
from loaders import TestDataset
from model import SiameseNetwork
from constants import LIGHTNING_LOGS, LABEL_DICT

VERSION_NUMBER = 30


def get_config(version_number: int):
    """Builds a config object using the version number, we can
    eye ball the best version by montoring the lightning logs.

    Params:
    -------
    version_number: int

    Returns:
    --------
    config: Config"""
    config_path = Path(LIGHTNING_LOGS / f"version_{version_number}/configs.json")
    return Config.build(config_path=config_path)


def best_rect(m: int) -> Tuple[int, int]:
    """Returns n_rows and n_cols giving the number of images we
    want to plot. (super handy)

    Params:
    -------
    m: int

    Returns:
    --------
    prod: Tuple[int, int]"""
    low = int(np.floor(np.sqrt(m)))
    high = int(np.ceil(np.sqrt(m)))
    prods = [(low, low), (low, low + 1), (high, high), (high, high + 1)]
    for i, prod in enumerate(prods):
        if prod[0] * prod[1] >= m:
            return prod
    raise ValueError("Not possible!")


def img_similarity_search(
    config: Config,
    version_number: int,
    input_img: Tensor,
    label: int,
    dataset: TestDataset,
    num_show_similar: int = 5,
):
    """Extracts and plots 'n' relevant images from a dataset where
    n is given by num_show_similar, it also takes a version number
    parameter specifying which trained model to pick.

    It would be ideal if we could pass num_show_similar a value such
    that we could display the images as a grid to avoid any empty blocks
    E.g. num_show_similar = 5
    total_imgs = (num_show_similar + input_img) = (5 + 1) = 6
    so we can get a 2 x 3 grid

    similarly, if num_show_similar = 8
    then, we can get a 3 x 3 grid

    Params:
    -------
    config: Config

    version_number: int

    dataset: TestDataset

    num_show_similar: int

    Returns:
    --------
    None"""
    trained_model = SiameseNetwork.load_from_checkpoint(
        Path(LIGHTNING_LOGS / f"version_{version_number}/checkpoints/last.ckpt"),
        config=config,
    )

    top_k_similar_images, similarity_scores = trained_model.find_similar_images(
        input_img, dataset, top_k=num_show_similar
    )
    plot_similar_images(
        input_img, label, dataset, top_k_similar_images, similarity_scores
    )


def plot_similar_images(
    input_img: Tensor,
    label: int,
    test_dataset: TestDataset,
    similar_img_indices: List[int],
    similarity_scores: List[float],
) -> None:
    """Plotting function given an input_image, dataset, similar image indices
    and similarity scores.

    Params:
    -------
    input_image: Tensor

    test_dataset: TestDataset

    similarity_image_indices: List[int]

    similarity_scores: List[float]

    Returns:
    --------
    None"""
    n_rows, n_cols = best_rect(len(similar_img_indices) + 1)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5))
    axes = axes.flatten()
    # Plot the input image
    axes[0].imshow(input_img.permute(1, 2, 0))
    axes[0].set_title(f"Input Image: {label} ({LABEL_DICT[str(label)]})")
    axes[0].axis("off")

    # Plot the similar images
    for i, idx in enumerate(similar_img_indices):
        image, label = test_dataset[idx]  # Extract image and label
        axes[i + 1].imshow(image.permute(1, 2, 0))  # Use the image part of the tuple
        axes[i + 1].set_title(
            f"Similarity Score: {similarity_scores[i]:.4f}\nLabel: {label}"
        )
        axes[i + 1].axis("off")

    # Show the plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # dummy test
    dataset = TestDataset()
    config = get_config(version_number=VERSION_NUMBER)

    # instead of this we can pass any random image as well
    num_images = len(dataset)
    idx = int(np.random.choice(num_images, 1)[0])
    img, label = dataset[idx]

    img_similarity_search(
        config=config,
        version_number=VERSION_NUMBER,
        input_img=img,
        label=label,
        dataset=dataset,
        num_show_similar=8,
    )
