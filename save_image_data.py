from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm

from constants import DATA_PATH, DATA, TRAIN_DIR, VAL_DIR, TEST_DIR


def save_image(data, idx, split):
    image_data = data["image"]
    image_label = data["label"]
    image_path = DATA / split / f"image_{idx}_label_{image_label}.png"

    image_data.save(image_path)


if __name__ == "__main__":
    train_dataset = load_dataset(DATA_PATH, split="train")
    val_dataset = load_dataset(DATA_PATH, split="validation")
    test_dataset = load_dataset(DATA_PATH, split="test")

    if not TRAIN_DIR.exists():
        TRAIN_DIR.mkdir(parents=True, exist_ok=True)

        for idx, data in tqdm(
            enumerate(train_dataset),
            total=len(train_dataset),
        ):
            save_image(data, idx, split="train")
        print(f"Train images saved at {TRAIN_DIR}")

    if not VAL_DIR.exists():
        VAL_DIR.mkdir(parents=True, exist_ok=True)
        for idx, data in tqdm(
            enumerate(val_dataset),
            total=len(val_dataset),
        ):
            save_image(data, idx, split="validation")
        print(f"Validation images saved at {VAL_DIR}")

    if not TEST_DIR.exists():
        TEST_DIR.mkdir(parents=True, exist_ok=True)
        for idx, data in tqdm(
            enumerate(test_dataset),
            total=len(test_dataset),
        ):
            save_image(data, idx, split="test")
        print(f"Test images saved at {TEST_DIR}")
