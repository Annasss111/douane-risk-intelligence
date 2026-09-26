from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .llm_service import chat_with_analyst, export_explanation, generate_explanation
from .model_service import fraud_model
from .network_service import network_service


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
app = FastAPI(title="Douane Risk Intelligence API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ExplanationRequest(BaseModel):
    record: dict[str, Any]


class ChatRequest(BaseModel):
    record: dict[str, Any]
    history: list[dict[str, str]] = []
    message: str


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": "catboost",
        "shap": "native_catboost",
        "threshold": fraud_model.threshold,
        "network": "community_graph_ready",
        "similarity": "knn_features",
        "llm": "openai_configured" if os.getenv("OPENAI_API_KEY") else "ollama_local",
        "ollama_model": os.getenv("OLLAMA_MODEL", "llama3:latest"),
    }


@app.post(
    "/api/score",
    responses={400: {"description": "Invalid CSV upload"}, 422: {"description": "Missing model features"}},
)
async def score_csv(file: Annotated[UploadFile, File(...)]) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a CSV file")
    try:
        content = await file.read()
        data = pd.read_csv(io.BytesIO(content))
        records = fraud_model.score(data)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Could not score file: {error}") from error

    network = network_service.analyze(data)
    return {
        "filename": file.filename,
        "summary": fraud_model.summarize(records),
        "records": records[:500],
        "network": {"summary": network["summary"], "nodes": network["nodes"][:500], "edges": network["edges"][:1500], "networks": network["networks"][:50]},
        "returned": min(len(records), 500),
        "truncated": len(records) > 500,
    }


@app.get("/api/demo", responses={404: {"description": "Demo dataset not found"}, 500: {"description": "Demo scoring failed"}})
def demo() -> dict[str, Any]:
    demo_path = ROOT / "df_syn_test_eng.csv"
    if not demo_path.exists():
        raise HTTPException(status_code=404, detail="Demo dataset not found")
    try:
        data = pd.read_csv(demo_path)
        records = fraud_model.score(data)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Could not load demo: {error}") from error
    network = network_service.analyze(data)
    return {
        "filename": demo_path.name,
        "summary": fraud_model.summarize(records),
        "records": records[:500],
        "network": {"summary": network["summary"], "nodes": network["nodes"][:500], "edges": network["edges"][:1500], "networks": network["networks"][:50]},
        "returned": min(len(records), 500),
        "truncated": len(records) > 500,
    }


@app.post("/api/explain")
def explain(request: ExplanationRequest) -> dict[str, str]:
    return generate_explanation(request.record)


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict[str, str]:
    if not request.message.strip():
        raise HTTPException(status_code=422, detail="Le message ne peut pas être vide")
    return chat_with_analyst(request.record, request.history, request.message.strip())


@app.post("/api/export")
async def export_csv(file: Annotated[UploadFile, File(...)]) -> StreamingResponse:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a CSV file")
    try:
        content = await file.read()
        data = pd.read_csv(io.BytesIO(content))
        records = fraud_model.score_for_export(data)
        enriched = data.copy()
        enriched["Fraude"] = ["Oui" if record["is_risky"] else "Non" for record in records]
        enriched["Explication"] = [export_explanation(record) for record in records]
        csv_content = enriched.to_csv(index=False).encode("utf-8-sig")
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Could not export file: {error}") from error
    output_name = f"{Path(file.filename).stem}_enrichi.csv"
    return StreamingResponse(
        io.BytesIO(csv_content),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{output_name}"'},
    )


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(FRONTEND / "index.html")


app.mount("/assets", StaticFiles(directory=FRONTEND / "assets"), name="assets")
