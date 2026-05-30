import argparse
import json

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.config import MODEL_DIR, MODEL_PATH, RANDOM_STATE, TRAIN_INFO_PATH, TRAIN_PATH
from src.data_utils import clean_text, load_dataset


def build_model(max_features: int, ngram_max: int, c_value: float) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=clean_text,
                    lowercase=False,
                    token_pattern=r"(?u)(?:#\w+|\b\w\w+\b)",
                    ngram_range=(1, ngram_max),
                    min_df=2,
                    max_df=0.95,
                    max_features=max_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                    solver="liblinear",
                ),
            ),
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a rumor detection baseline.")
    parser.add_argument("--train-path", default=str(TRAIN_PATH))
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    parser.add_argument("--max-features", type=int, default=30000)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--c-value", type=float, default=2.0)
    parser.add_argument("--cv-folds", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_df = load_dataset(args.train_path)

    model = build_model(
        max_features=args.max_features,
        ngram_max=args.ngram_max,
        c_value=args.c_value,
    )

    x_train = train_df["text"].tolist()
    y_train = train_df["label"].tolist()

    cv = StratifiedKFold(
        n_splits=args.cv_folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    cv_scores = cross_val_score(model, x_train, y_train, cv=cv, scoring="accuracy")

    model.fit(x_train, y_train)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_INFO_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_path)

    train_info = {
        "train_path": args.train_path,
        "model_path": args.model_path,
        "num_train_samples": int(len(train_df)),
        "label_distribution": {
            str(label): int(count)
            for label, count in train_df["label"].value_counts().sort_index().items()
        },
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "max_features": args.max_features,
        "ngram_max": args.ngram_max,
        "c_value": args.c_value,
    }

    TRAIN_INFO_PATH.write_text(
        json.dumps(train_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved model to {args.model_path}")
    print(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


if __name__ == "__main__":
    main()
