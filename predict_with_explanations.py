import argparse
import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool


MODEL_PATH = "catboost_fraud_model.cbm"
THRESHOLD_PATH = "catboost_threshold.json"
ID_COLUMN = "Declaration ID"


def json_safe_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def prepare_features(data, feature_names, categorical_features):
    missing_features = [feature for feature in feature_names if feature not in data.columns]
    if missing_features:
        raise ValueError(f"Missing model features: {missing_features}")

    prepared = data[feature_names].copy()
    for feature in categorical_features:
        prepared[feature] = prepared[feature].fillna("Unknown").astype(str)
    return prepared


def explain_predictions(model, data, prepared_data, probabilities, threshold, top_n=5):
    categorical_features = [
        model.feature_names_[index]
        for index in model.get_cat_feature_indices()
    ]
    prediction_pool = Pool(prepared_data, cat_features=categorical_features)
    shap_values = model.get_feature_importance(
        type="ShapValues",
        data=prediction_pool,
    )
    contributions = shap_values[:, :-1]
    base_values = shap_values[:, -1]
    predictions = probabilities >= threshold
    feature_names = model.feature_names_
    records = []

    for row_index, row_contributions in enumerate(contributions):
        factors = [
            {
                "feature": feature_name,
                "value": json_safe_value(data.iloc[row_index][feature_name]),
                "contribution": round(float(row_contributions[feature_index]), 6),
            }
            for feature_index, feature_name in enumerate(feature_names)
        ]
        risk_factors = sorted(
            (factor for factor in factors if factor["contribution"] > 0),
            key=lambda factor: factor["contribution"],
            reverse=True,
        )[:top_n]
        protective_factors = sorted(
            (factor for factor in factors if factor["contribution"] < 0),
            key=lambda factor: factor["contribution"],
        )[:top_n]

        records.append(
            {
                ID_COLUMN: data.iloc[row_index].get(ID_COLUMN, row_index),
                "fraud_probability": round(float(probabilities[row_index]), 6),
                "is_risky": bool(predictions[row_index]),
                "threshold": round(float(threshold), 6),
                "base_value": round(float(base_values[row_index]), 6),
                "risk_factors": risk_factors,
                "protective_factors": protective_factors,
            }
        )
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Predict customs fraud and generate local SHAP explanations."
    )
    parser.add_argument("input_csv", help="CSV containing transactions to score")
    parser.add_argument(
        "--output",
        default="predictions_with_explanations.json",
        help="JSON output path",
    )
    args = parser.parse_args()

    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    with open(THRESHOLD_PATH, encoding="utf-8") as threshold_file:
        threshold = float(json.load(threshold_file)["threshold"])

    data = pd.read_csv(args.input_csv)
    feature_names = model.feature_names_
    categorical_features = [
        feature_names[index]
        for index in model.get_cat_feature_indices()
    ]
    prepared_data = prepare_features(data, feature_names, categorical_features)
    prediction_pool = Pool(prepared_data, cat_features=categorical_features)
    probabilities = model.predict_proba(prediction_pool)[:, 1]
    records = explain_predictions(
        model,
        data,
        prepared_data,
        probabilities,
        threshold,
    )

    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(records, output_file, ensure_ascii=False, default=str, indent=2)

    print(f"Scored {len(records)} transactions")
    print(f"Risky transactions: {sum(record['is_risky'] for record in records)}")
    print(f"Explanations saved to: {args.output}")


if __name__ == "__main__":
    main()
