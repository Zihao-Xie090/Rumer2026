import html
import re
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"id", "text", "label", "event"}


def clean_text(text: object) -> str:
    """Normalize tweet text while keeping useful social-media signals."""
    if not isinstance(text, str):
        return ""

    text = html.unescape(text)
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"@\w+", " USER ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)
    df["event"] = df["event"].astype(int)
    df["text_clean"] = df["text"].map(clean_text)
    return df
