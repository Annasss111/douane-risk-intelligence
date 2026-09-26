import pandas as pd
import numpy as np
import time
import json
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from catboost import CatBoostClassifier, Pool

print(" Démarrage du pipeline de détection de fraude douanière")

# 1. Chargement des données
print(" Chargement des fichiers CSV")
df_train = pd.read_csv("df_syn_train_eng.csv")
df_valid = pd.read_csv("df_syn_valid_eng.csv")
df_test  = pd.read_csv("df_syn_test_eng.csv")

print(f"Taille Train : {df_train.shape} | Valid : {df_valid.shape} | Test : {df_test.shape}")

# 2. Préparation
TARGET = 'Fraud'
ID_COLUMN = 'Declaration ID'

# On enlève les ID qui ne servent à rien pour l'entrainement ML pur
# (Mais ton collègue GNN en aura besoin !)
cols_to_drop = [TARGET, 'Critical Fraud', ID_COLUMN, 'Date']
features = [c for c in df_train.columns if c not in cols_to_drop]

X_train = df_train[features].copy()
y_train = df_train[TARGET].copy()

X_valid = df_valid[features].copy()
y_valid = df_valid[TARGET].copy()

X_test  = df_test[features].copy()
y_test  = df_test[TARGET].copy()

# 3. Traitement des variables catégoriques
# CatBoost adore savoir quelles colonnes sont des catégories
cat_features_names = X_train.select_dtypes(include=['object', 'string']).columns.tolist()

# Harmoniser les catégories entre train, validation et test.
for col in cat_features_names:
    # 1. Remplacer les valeurs manquantes d'origine par le texte "Unknown"
    X_train[col] = X_train[col].fillna("Unknown").astype(str)
    X_valid[col] = X_valid[col].fillna("Unknown").astype(str)
    X_test[col]  = X_test[col].fillna("Unknown").astype(str)
    
    # 2. Récupérer les catégories connues dans le Train
    train_categories = X_train[col].unique().tolist()
    
    # 3. Ajouter "Unknown" à la liste des catégories autorisées (s'il n'y est pas)
    if "Unknown" not in train_categories:
        train_categories.append("Unknown")
        
    # 4. Magie : Remplacer toute catégorie inconnue dans Valid et Test par "Unknown"
    X_valid.loc[~X_valid[col].isin(train_categories), col] = "Unknown"
    X_test.loc[~X_test[col].isin(train_categories), col] = "Unknown"
    
    # 5. Convertir tout en type Category propre (sans NaN !)
    cat_type = pd.CategoricalDtype(categories=train_categories, ordered=False)
    X_train[col] = X_train[col].astype(cat_type)
    X_valid[col] = X_valid[col].astype(cat_type)
    X_test[col]  = X_test[col].astype(cat_type)

print(f"📊 Features prêtes : {len(features)} colonnes.")
print(f"📌 Colonnes Catégoriques gérées : {cat_features_names}")


# ==========================================
# 4. ENTRAINEMENT CATBOOST
# ==========================================

# Calcul du ratio exact (pour info, c'est environ 3.6)
ratio_exact = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
print(f"Ratio naturel (Classe 0 / Classe 1) : {ratio_exact:.2f}")

print("\n🚀 Entrainement CatBoost (PRO LEVEL) sur GPU...")
start_time = time.time()

cat_model = CatBoostClassifier(
    iterations=1500,             # Zidna l'wa9t chwaya
    learning_rate=0.03,
    depth=8,                     # <-- Zidna l'profondeur (Modèle adhka)
    l2_leaf_reg=5,               # <-- Zedna l'regularisation bch ma yaamalch overfitting
    task_type="GPU",
    cat_features=cat_features_names,
    auto_class_weights='SqrtBalanced', # <-- Poids optimal automatique (Racine carrée)
    eval_metric='AUC',
    loss_function='Logloss',
    random_state=42
)

cat_model.fit(
    X_train, y_train,
    eval_set=(X_valid, y_valid),
    early_stopping_rounds=70,    # Khalineh yasber 70 iterations 9bal ma y9oss
    verbose=False
)



cat_time = time.time() - start_time
print(f"✅ CatBoost PRO terminé en {cat_time:.2f} secondes.")





# ==========================================
# 5. SELECTION DU SEUIL ET EVALUATION
# ==========================================
def metrics_at_threshold(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
    }


valid_probs = cat_model.predict_proba(X_valid)[:, 1]
thresholds = np.linspace(0.05, 0.95, 181)
valid_results = [
    metrics_at_threshold(y_valid, valid_probs, threshold)
    for threshold in thresholds
]

# Le recall fraude est prioritaire. On conserve toutefois une précision minimale
# pour éviter un seuil qui déclencherait trop de fausses alertes.
MIN_PRECISION = 0.40
eligible_results = [
    result for result in valid_results
    if result["precision"] >= MIN_PRECISION
]
best_valid = max(
    eligible_results or valid_results,
    key=lambda result: (result["recall"], result["precision"]),
)
cat_probs = cat_model.predict_proba(X_test)[:, 1]

print("\n📈 EVALUATION CATBOOST SUR LE TEST SET")
print(f"Seuil choisi sur validation (recall prioritaire, precision >= {MIN_PRECISION:.2f}) : {best_valid['threshold']:.3f}")
print(f"Validation | accuracy={best_valid['accuracy']:.4f} | precision={best_valid['precision']:.4f} | recall={best_valid['recall']:.4f} | F1={best_valid['f1']:.4f}")
print(f"Test ROC-AUC : {roc_auc_score(y_test, cat_probs):.4f}")

for result_name, threshold in (("Seuil 0.500", 0.5), ("Seuil optimise", best_valid["threshold"])):
    test_result = metrics_at_threshold(y_test, cat_probs, threshold)
    test_predictions = (cat_probs >= threshold).astype(int)
    print(f"\n--- {result_name} ---")
    print(
        "accuracy={accuracy:.4f} | precision={precision:.4f} | "
        "recall={recall:.4f} | F1={f1:.4f}".format(**test_result)
    )
    print(classification_report(y_test, test_predictions, zero_division=0))


# ==========================================
# 6. EXPLICATIONS SHAP DES TRANSACTIONS
# ==========================================
def json_safe_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def build_shap_explanations(model, prepared_data, original_data, probabilities, threshold, top_n=5):
    """Create local explanations using CatBoost's native SHAP implementation."""
    shap_pool = Pool(prepared_data, cat_features=cat_features_names)
    shap_values = model.get_feature_importance(
        type="ShapValues",
        data=shap_pool,
    )
    contributions = shap_values[:, :-1]
    base_values = shap_values[:, -1]
    predictions = (probabilities >= threshold).astype(int)
    records = []

    for row_index, row_contributions in enumerate(contributions):
        feature_details = []
        for feature_index, feature_name in enumerate(features):
            feature_details.append({
                "feature": feature_name,
                "value": json_safe_value(original_data.iloc[row_index][feature_name]),
                "contribution": round(float(row_contributions[feature_index]), 6),
            })

        risk_factors = sorted(
            (item for item in feature_details if item["contribution"] > 0),
            key=lambda item: item["contribution"],
            reverse=True,
        )[:top_n]
        protective_factors = sorted(
            (item for item in feature_details if item["contribution"] < 0),
            key=lambda item: item["contribution"],
        )[:top_n]

        record = {
            ID_COLUMN: original_data.iloc[row_index].get(ID_COLUMN, row_index),
            "fraud_probability": round(float(probabilities[row_index]), 6),
            "is_risky": bool(predictions[row_index]),
            "threshold": round(float(threshold), 6),
            "base_value": round(float(base_values[row_index]), 6),
            "risk_factors": risk_factors,
            "protective_factors": protective_factors,
        }
        records.append(record)

    return records, contributions


model_path = "catboost_fraud_model.cbm"
cat_model.save_model(model_path)

shap_records, shap_contributions = build_shap_explanations(
    cat_model,
    X_test,
    df_test,
    cat_probs,
    best_valid["threshold"],
)

explanation_df = pd.DataFrame({
    ID_COLUMN: [record[ID_COLUMN] for record in shap_records],
    "fraud_probability": [record["fraud_probability"] for record in shap_records],
    "is_risky": [record["is_risky"] for record in shap_records],
    "threshold": [record["threshold"] for record in shap_records],
    "base_value": [record["base_value"] for record in shap_records],
    "risk_factors": [json.dumps(record["risk_factors"], default=str) for record in shap_records],
    "protective_factors": [json.dumps(record["protective_factors"], default=str) for record in shap_records],
})
for feature_index, feature_name in enumerate(features):
    explanation_df[f"shap_{feature_name}"] = shap_contributions[:, feature_index]

explanation_df.to_csv("shap_explanations_test.csv", index=False)
with open("shap_explanations_test.json", "w", encoding="utf-8") as explanation_file:
    json.dump(shap_records, explanation_file, ensure_ascii=False, default=str, indent=2)

with open("catboost_threshold.json", "w", encoding="utf-8") as threshold_file:
    json.dump({"threshold": float(best_valid["threshold"])}, threshold_file, indent=2)

print("\n🔎 Couche SHAP générée")
print(f"✅ Modèle sauvegardé : {model_path}")
print("✅ Explications CSV : shap_explanations_test.csv")
print("✅ Explications JSON : shap_explanations_test.json")
print("ℹ️ Les contributions positives augmentent le risque de fraude; les négatives le diminuent.")


