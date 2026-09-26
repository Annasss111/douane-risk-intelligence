from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.neighbors import NearestNeighbors

from .network_service import network_service


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "catboost_fraud_model.cbm"
THRESHOLD_PATH = ROOT / "catboost_threshold.json"
ID_COLUMN = "Declaration ID"


class FraudModelService:
    def __init__(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        self.model = CatBoostClassifier()
        self.model.load_model(str(MODEL_PATH))
        self.threshold = self._load_threshold()
        self.feature_names = list(self.model.feature_names_)
        self.categorical_features = [
            self.feature_names[index]
            for index in self.model.get_cat_feature_indices()
        ]

    @staticmethod
    def _load_threshold() -> float:
        with THRESHOLD_PATH.open(encoding="utf-8") as threshold_file:
            return float(json.load(threshold_file)["threshold"])

    @staticmethod
    def _safe_value(value: Any) -> Any:
        if pd.isna(value):
            return None
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        return value

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        missing = [feature for feature in self.feature_names if feature not in data.columns]
        if missing:
            raise ValueError(f"Missing model features: {missing}")

        prepared = data[self.feature_names].copy()
        for feature in self.categorical_features:
            prepared[feature] = prepared[feature].fillna("Unknown").astype(str)
        return prepared

    def score(self, data: pd.DataFrame, top_n: int = 5) -> list[dict[str, Any]]:
        prepared = self.prepare_features(data)
        pool = Pool(prepared, cat_features=self.categorical_features)
        probabilities = self.model.predict_proba(pool)[:, 1]
        shap_values = self.model.get_feature_importance(type="ShapValues", data=pool)
        contributions = shap_values[:, :-1]
        base_values = shap_values[:, -1]
        network_analysis = network_service.analyze(data)
        similar_rows = self._similar_rows(data, probabilities)
        records = []

        for row_index, probability in enumerate(probabilities):
            factors = [
                {
                    "feature": feature_name,
                    "value": self._safe_value(data.iloc[row_index][feature_name]),
                    "contribution": round(float(contributions[row_index][feature_index]), 6),
                }
                for feature_index, feature_name in enumerate(self.feature_names)
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
            risk_level = self._risk_level(float(probability))
            network_data = network_analysis["row_network"].get(row_index, {})
            item_price = self._numeric_value(data.iloc[row_index].get("Item Price"))
            estimated_recovery = item_price * float(probability) * 0.35
            records.append(
                {
                    "row_index": row_index,
                    "declaration_id": self._safe_value(data.iloc[row_index].get(ID_COLUMN, row_index)),
                    "fraud_probability": round(float(probability), 6),
                    "is_risky": bool(probability >= self.threshold),
                    "risk_level": risk_level,
                    "threshold": round(self.threshold, 6),
                    "base_value": round(float(base_values[row_index]), 6),
                    "network_risk": network_data.get("network_risk", 0.0),
                    "network_id": network_data.get("network_id"),
                    "risk_sources": ["fraud_model", "shap_explanation"] + (["network_graph"] if network_data else []),
                    "estimated_recovery_tnd": round(estimated_recovery, 2),
                    "roi_score": round(min(100.0, float(probability) * 100 * (1 + np.log1p(max(item_price, 0)) / 10)), 2),
                    "similar_transactions": similar_rows[row_index],
                    "risk_factors": risk_factors,
                    "protective_factors": protective_factors,
                }
            )
        return records

    def score_for_export(self, data: pd.DataFrame) -> list[dict[str, Any]]:
        """Fast export path: only calculate the fields needed in the CSV."""
        prepared = self.prepare_features(data)
        pool = Pool(prepared, cat_features=self.categorical_features)
        probabilities = self.model.predict_proba(pool)[:, 1]
        return [
            {
                "is_risky": bool(probability >= self.threshold),
                "risk_level": self._risk_level(float(probability)),
                "fraud_probability": float(probability),
                "risk_factors": [],
            }
            for probability in probabilities
        ]

    def _similar_rows(self, data: pd.DataFrame, probabilities: np.ndarray) -> dict[int, list[dict[str, Any]]]:
        if len(data) < 2:
            return {0: []} if len(data) else {}
        matrix = []
        for feature in self.feature_names:
            column = data[feature]
            if feature in self.categorical_features:
                values = pd.factorize(column.fillna("Unknown").astype(str))[0].astype(float)
            else:
                values = pd.to_numeric(column, errors="coerce").fillna(pd.to_numeric(column, errors="coerce").median()).fillna(0).to_numpy(dtype=float)
            scale = float(np.std(values)) or 1.0
            matrix.append((values - float(np.mean(values))) / scale)
        vectors = np.column_stack(matrix)
        neighbors = NearestNeighbors(n_neighbors=min(4, len(data)), metric="euclidean").fit(vectors)
        distances, indices = neighbors.kneighbors(vectors)
        result = {}
        for row_index, (row_distances, row_indices) in enumerate(zip(distances, indices)):
            matches = []
            for distance, match_index in zip(row_distances[1:], row_indices[1:]):
                comparison = []
                for feature_index, feature_name in enumerate(self.feature_names):
                    current_value = self._safe_value(data.iloc[row_index].get(feature_name))
                    match_value = self._safe_value(data.iloc[match_index].get(feature_name))
                    feature_distance = abs(float(vectors[row_index, feature_index] - vectors[match_index, feature_index]))
                    comparison.append({
                        "feature": feature_name,
                        "current_value": current_value,
                        "similar_value": match_value,
                        "similarity": round(1 / (1 + feature_distance), 4),
                    })
                comparison.sort(key=lambda item: item["similarity"], reverse=True)
                match_probability = float(probabilities[match_index])
                matches.append({
                    "row_index": int(match_index),
                    "declaration_id": self._safe_value(data.iloc[match_index].get(ID_COLUMN, match_index)),
                    "fraud_probability": round(match_probability, 6),
                    "risk_level": self._risk_level(match_probability),
                    "similarity": round(1 / (1 + float(distance)), 4),
                    "comparison": comparison[:6],
                })
            result[row_index] = matches
        return result

    @staticmethod
    def _numeric_value(value: Any) -> float:
        try:
            return float(value) if not pd.isna(value) else 0.0
        except (TypeError, ValueError):
            return 0.0

    def _risk_level(self, probability: float) -> str:
        if probability >= 0.75:
            return "critical"
        if probability >= self.threshold:
            return "high"
        if probability >= max(self.threshold * 0.7, 0.25):
            return "watch"
        return "low"

    def summarize(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        probabilities = [record["fraud_probability"] for record in records]
        return {
            "total": len(records),
            "risky": sum(record["is_risky"] for record in records),
            "critical": sum(record["risk_level"] == "critical" for record in records),
            "high": sum(record["risk_level"] == "high" for record in records),
            "watch": sum(record["risk_level"] == "watch" for record in records),
            "average_probability": round(float(np.mean(probabilities)), 4) if probabilities else 0,
            "threshold": round(self.threshold, 6),
            "networked": sum(1 for record in records if record.get("network_id")),
            "estimated_recovery_tnd": round(sum(record.get("estimated_recovery_tnd", 0) for record in records), 2),
            "vnn_status": "community_graph_ready",
        }


fraud_model = FraudModelService()
