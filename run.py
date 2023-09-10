from torch.utils.data import DataLoader
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from configs import Config
from loaders import SnacksDataset, TestDataset
from train import get_loaders, train_siamese

if __name__ == "__main__":
    # Ideally we want to using a 4x2 or 6x2 grid for lr and wd
    # lr = [1e-1, 1e-2, 1e-3, 1e-4]
    # wd = [1e-2, 1e-4]
    # and check with different kinds of backbone - (image encoder)
    # backbone = ["resnext", "efficientnet", "mobilenet", "wide_resnet"]

    grid_dicts = list(
        ParameterGrid(
            dict(
                backbone=[
                    # "resnext",
                    # "efficientnet",
                    "mobilenet",
                    # "wide_resnet",
                ],
                augment=[False, True],
                lr=[3e-4],
                wd=[1e-2],
                epochs=[30],
            )
        )
    )

    configs = [Config(**grid_dict) for grid_dict in grid_dicts]
    pbar = tqdm(configs)
    for config in pbar:
        print(config)
        train_loader, val_loader, test_dataset = get_loaders(config=config)

        train_siamese(
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
        )
