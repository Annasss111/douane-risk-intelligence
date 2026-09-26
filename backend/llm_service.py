from __future__ import annotations

import json
import os
import urllib.request
from typing import Any


SYSTEM_PROMPT = """You are a customs fraud analyst assistant for Tunisian customs.
Explain model alerts clearly and cautiously in French. Never claim that a transaction is
fraudulent with certainty: the model identifies risk signals for human review.
Use only the supplied probability, severity, SHAP factors, and KNN similar cases. Mention
that positive contributions increase the model's fraud score and negative contributions
reduce it. Keep the response structured in: verdict, main reasons, a concrete comparison
with one similar case of the same severity when available, and recommended human checks.
Be concise: answer only in French, in at most 6 short bullet points and 100 words. Never repeat
the complete input data."""


def _severity(record: dict[str, Any]) -> str:
    if record.get("risk_level"):
        return str(record["risk_level"])
    probability = float(record.get("fraud_probability", 0))
    threshold = float(record.get("threshold", 0.36))
    if probability >= 0.75:
        return "critical"
    if probability >= threshold:
        return "high"
    if probability >= max(threshold * 0.7, 0.25):
        return "watch"
    return "low"


def _same_severity_case(record: dict[str, Any]) -> dict[str, Any] | None:
    severity = _severity(record)
    for case in record.get("similar_transactions", []):
        if _severity(case) == severity:
            return case
    return None


def _rule_based_explanation(record: dict[str, Any]) -> str:
    probability = float(record.get("fraud_probability", 0)) * 100
    level = record.get("risk_level", "watch")
    factors = record.get("risk_factors", [])
    reasons = "; ".join(
        f"{factor['feature']}={factor['value']} (+{factor['contribution']:.3f})"
        for factor in factors[:3]
    ) or "aucun facteur dominant disponible"
    similar_case = _same_severity_case(record)
    comparison = (
        f" Exemple concret: la déclaration {similar_case.get('declaration_id', 'similaire')} "
        f"présente une gravité comparable ({float(similar_case.get('fraud_probability', 0)) * 100:.1f}%) "
        f"et une similarité KNN de {float(similar_case.get('similarity', 0)) * 100:.1f}%."
        if similar_case else " Aucun cas KNN de même gravité n'est disponible dans ce périmètre."
    )
    return (
        f"Niveau {level.upper()} avec une probabilité de risque de {probability:.1f}%. "
        f"Les signaux principaux sont: {reasons}. "
        + comparison
        + " "
        "Cette alerte est une aide à la décision et doit être vérifiée par un agent, "
        "notamment avec les documents commerciaux, la valeur déclarée et l'origine."
    )


def export_explanation(record: dict[str, Any]) -> str:
    """Return a fast deterministic explanation suitable for every CSV row."""
    probability = float(record.get("fraud_probability", 0)) * 100
    level = _severity(record).upper()
    factors = record.get("risk_factors", [])
    reasons = ", ".join(str(factor.get("feature")) for factor in factors[:2]) or "signaux du modèle"
    return f"Gravité {level}; probabilité {probability:.1f}%; facteurs principaux: {reasons}. Vérification humaine recommandée."


def _context(record: dict[str, Any]) -> dict[str, Any]:
    similar_case = _same_severity_case(record)
    return {
        "declaration_id": record.get("declaration_id"),
        "probability": record.get("fraud_probability"),
        "severity": _severity(record),
        "risk_factors": record.get("risk_factors", []),
        "protective_factors": record.get("protective_factors", []),
        "similar_case_same_severity": similar_case,
    }


def _ollama_chat(messages: list[dict[str, str]]) -> str:
    model = os.getenv("OLLAMA_MODEL", "llama3:latest")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    request = urllib.request.Request(
        f"{base_url}/api/chat",
        data=json.dumps({"model": model, "messages": messages, "stream": False, "options": {"temperature": 0.2, "num_predict": 180}}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        result = json.loads(response.read().decode("utf-8"))
    text = str(result["message"]["content"])
    words = text.split()
    if len(words) > 110:
        text = " ".join(words[:110]).rstrip(".,;:") + "..."
    return text


def generate_explanation(record: dict[str, Any]) -> dict[str, str]:
    context = _context(record)
    ollama_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
    ]
    try:
        return {"source": "ollama", "text": _ollama_chat(ollama_messages)}
    except Exception:
        pass

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"source": "rule_based", "text": _rule_based_explanation(record)}

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        **context,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            result = json.loads(response.read().decode("utf-8"))
        text = result["choices"][0]["message"]["content"]
        return {"source": "llm", "text": text}
    except Exception:
        return {"source": "rule_based_fallback", "text": _rule_based_explanation(record)}


def chat_with_analyst(record: dict[str, Any], history: list[dict[str, str]], message: str) -> dict[str, str]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT + " Réponds directement à la question de l'analyste."}]
    messages.append({"role": "system", "content": f"Dossier analysé: {json.dumps(_context(record), ensure_ascii=False)}"})
    messages.extend(history[-10:])
    messages.append({"role": "user", "content": message})
    try:
        return {"source": "ollama", "text": _ollama_chat(messages)}
    except Exception:
        return {
            "source": "rule_based_fallback",
            "text": "Je peux répondre sur les facteurs SHAP, la gravité et le cas KNN comparable. "
            + _rule_based_explanation(record),
        }
