from __future__ import annotations

from dataclasses import dataclass

import numpy as np


LABEL_NAMES = {
    0: "非谣言",
    1: "谣言",
}


@dataclass
class Explanation:
    prediction: int
    label_name: str
    probability: float | None
    top_features: list[str]
    explanation: str

    def to_dict(self) -> dict:
        return {
            "prediction": self.prediction,
            "label_name": self.label_name,
            "probability": self.probability,
            "top_features": self.top_features,
            "explanation": self.explanation,
        }


def _get_probability(model, text: str, prediction: int) -> float | None:
    if not hasattr(model, "predict_proba"):
        return None
    proba = model.predict_proba([text])[0]
    class_to_index = {int(label): idx for idx, label in enumerate(model.classes_)}
    return float(proba[class_to_index[prediction]])


def _top_contributing_features(model, text: str, prediction: int, top_k: int) -> list[str]:
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["clf"]

    vector = vectorizer.transform([text])
    if vector.nnz == 0:
        return []

    feature_names = np.asarray(vectorizer.get_feature_names_out())
    coef = classifier.coef_[0]

    # Positive scores support label 1. For label 0, reverse the direction.
    direction = 1 if prediction == 1 else -1
    scores = vector.multiply(coef * direction).toarray()[0]
    nonzero = vector.nonzero()[1]
    ranked = sorted(nonzero, key=lambda index: scores[index], reverse=True)

    features: list[str] = []
    for index in ranked:
        if scores[index] <= 0:
            continue
        feature = str(feature_names[index])
        if feature not in features:
            features.append(feature)
        if len(features) >= top_k:
            break

    return features


def build_explanation(model, text: str, top_k: int = 5) -> Explanation:
    prediction = int(model.predict([text])[0])
    label_name = LABEL_NAMES[prediction]
    probability = _get_probability(model, text, prediction)
    top_features = _top_contributing_features(model, text, prediction, top_k)

    if top_features:
        features_text = "、".join(f"“{feature}”" for feature in top_features)
        if prediction == 1:
            explanation = (
                f"模型判断该文本为谣言，主要依据是文本中的 {features_text} "
                "等词语或短语在训练集中更常与谣言样本同时出现，并且这些特征对当前预测的正向贡献较高。"
            )
        else:
            explanation = (
                f"模型判断该文本为非谣言，主要依据是文本中的 {features_text} "
                "等词语或短语在训练集中更常与非谣言样本同时出现，并且这些特征对当前预测的正向贡献较高。"
            )
    else:
        explanation = (
            f"模型判断该文本为{label_name}。当前文本中的有效 TF-IDF 特征较少，"
            "因此判断主要来自整体文本表示与训练集中样本的相似性。"
        )

    return Explanation(
        prediction=prediction,
        label_name=label_name,
        probability=probability,
        top_features=top_features,
        explanation=explanation,
    )
