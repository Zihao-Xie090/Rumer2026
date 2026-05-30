import argparse
import json

import joblib

from src.config import MODEL_PATH
from src.explain import build_explanation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict one rumor label with explanation.")
    parser.add_argument("--text", required=True, help="Input tweet text.")
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = joblib.load(args.model_path)
    result = build_explanation(model, args.text).to_dict()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
