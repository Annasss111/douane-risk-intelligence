# Douane Risk Intelligence

Plateforme de priorisation des contrôles douaniers tunisiens, combinant scoring transactionnel, explicabilité, intelligence réseau et assistant analyste local.

## Architecture

- `catboost_fraud_model.cbm`: modèle CatBoost entraîné.
- `catboost_threshold.json`: seuil opérationnel choisi sur validation.
- `backend/model_service.py`: scoring, préparation des catégories et SHAP local.
- `backend/network_service.py`: graphe multi-entités, communautés Louvain pondérées, cohésion et groupes suspects.
- `backend/llm_service.py`: explication LLM via Ollama/Llama 3 avec fallback déterministe.
- `backend/main.py`: API FastAPI.
- `frontend/`: dashboard de supervision.
- `predict_with_explanations.py`: script batch sans interface.

Le dashboard combine quatre signaux : classification transactionnelle, SHAP local, graphe communautaire et cas similaires KNN. Chaque dossier contient aussi une estimation de récupération et un score ROI.

## Démarrage

Dans PowerShell, depuis `C:\Hackathon\hackathon`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn backend.main:app --reload
```

Puis ouvrir http://127.0.0.1:8000.

L'API expose `GET /api/health`, `GET /api/demo`, `POST /api/score`, `POST /api/explain`, `POST /api/chat` et `POST /api/export`. Le frontend est servi directement par FastAPI; Streamlit n'est pas requis.

Le bouton `Charger l'exemple` utilise `df_syn_test_eng.csv`. Pour analyser un nouveau fichier, importer un CSV contenant les features du modèle. La colonne `Fraud` n'est pas nécessaire.

## Stack

- Backend : Python, FastAPI, Uvicorn, CatBoost, NumPy, Pandas, NetworkX, scikit-learn.
- Modèle fraude : CatBoost avec seuil opérationnel `0.36`.
- Explicabilité : valeurs SHAP natives CatBoost.
- Similarité : KNN sur les features du modèle.
- Réseaux : graphe multi-entités et communautés Louvain pondérées.
- Assistant : Ollama local avec `llama3:latest`, fallback déterministe si Ollama est indisponible.
- Frontend : HTML, CSS et JavaScript vanilla, servi par FastAPI.

## LLM

Sans `OPENAI_API_KEY`, la plateforme génère une justification locale à partir des facteurs SHAP. Pour activer un LLM compatible OpenAI, configurer les variables de `.env.example` dans l'environnement avant de lancer Uvicorn.

```powershell
$env:OPENAI_API_KEY = "..."
$env:OPENAI_MODEL = "gpt-4o-mini"
uvicorn backend.main:app --reload
```

Le LLM reçoit uniquement le contexte du dossier, les facteurs SHAP et le cas KNN comparable. Il ne décide pas du risque et ne remplace pas la validation humaine.

## Résultats de référence

- Dataset de démonstration : `8 481` déclarations.
- Seuil de décision : `0.36`.
- Réseau : `49 500` connexions sur le dataset complet.
- Export enrichi : colonnes `Fraude` et `Explication`.
