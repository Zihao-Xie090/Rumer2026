from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT_DIR / "train.csv"
VAL_PATH = ROOT_DIR / "val.csv"

MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"

MODEL_PATH = MODEL_DIR / "tfidf_lr.joblib"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
VAL_PREDICTIONS_PATH = OUTPUT_DIR / "val_predictions.csv"
TRAIN_INFO_PATH = OUTPUT_DIR / "train_info.json"

RANDOM_STATE = 42
