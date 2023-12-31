from pathlib import Path
from datasets import load_dataset


def ensure_dir(path: Path) -> Path:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise FileExistsError(f"Specified path {path} is not a dir")
    return path


ROOT = Path(__file__).resolve().parent
DATA_PATH = "Matthijs/snacks"
DATA = ensure_dir(ROOT / "data")
TRAIN_DIR = DATA / "train"
VAL_DIR = DATA / "validation"
TEST_DIR = DATA / "test"
LIGHTNING_LOGS = ROOT / "lightning_logs"

LABEL_DICT = {
    "0": "apple",
    "1": "banana",
    "2": "cake",
    "3": "candy",
    "4": "carrot",
    "5": "cookie",
    "6": "doughnut",
    "7": "grape",
    "8": "hot dog",
    "9": "ice cream",
    "10": "juice",
    "11": "muffin",
    "12": "orange",
    "13": "pineapple",
    "14": "popcorn",
    "15": "pretzel",
    "16": "salad",
    "17": "strawberry",
    "18": "waffle",
    "19": "watermelon",
}
