import argparse
import json

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from tqdm import tqdm

from src.config import METRICS_PATH, MODEL_PATH, OUTPUT_DIR, VAL_PATH, VAL_PREDICTIONS_PATH
from src.data_utils import load_dataset
from src.explain import build_explanation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the rumor detection model.")
    parser.add_argument("--val-path", default=str(VAL_PATH))
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    parser.add_argument("--metrics-path", default=str(METRICS_PATH))
    parser.add_argument("--predictions-path", default=str(VAL_PREDICTIONS_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    val_df = load_dataset(args.val_path)
    model = joblib.load(args.model_path)

    x_val = val_df["text"].tolist()
    y_true = val_df["label"].tolist()
    y_pred = model.predict(x_val)

    metrics = {
        "num_val_samples": int(len(val_df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).astype(int).tolist(),
    }

    probabilities = []
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_val)
        class_to_index = {int(label): idx for idx, label in enumerate(model.classes_)}
        probabilities = [
            float(row[class_to_index[int(pred)]]) for row, pred in zip(proba, y_pred)
        ]
    else:
        probabilities = [None] * len(y_pred)

    explanations = [
        build_explanation(model, text).to_dict()
        for text in tqdm(x_val, desc="Generating explanations")
    ]

    output_df = pd.DataFrame(
        {
            "id": val_df["id"],
            "text": val_df["text"],
            "label": y_true,
            "prediction": y_pred.astype(int),
            "probability": probabilities,
            "top_features": ["; ".join(item["top_features"]) for item in explanations],
            "explanation": [item["explanation"] for item in explanations],
        }
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    output_df.to_csv(args.predictions_path, index=False, encoding="utf-8-sig")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved metrics to {args.metrics_path}")
    print(f"Saved predictions to {args.predictions_path}")


if __name__ == "__main__":
    main()
