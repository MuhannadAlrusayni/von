"""FastAPI server for Hop implementing TypeSafe-compatible HTTP endpoints."""

import os
from typing import Any, Dict, Optional
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .engine import HopEngine
from .types import SystemOneResponse

app = FastAPI(
    title="Hop Decision Server",
    description="Drop-in open source System One decision engine",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SystemOneRequest(BaseModel):
    model: str = Field(default="hop-latest")
    state: Any = Field(..., description="State object, string, or array to evaluate")
    questions: Dict[str, Dict[str, Any]] = Field(..., description="Dict of question definitions")


@app.get("/")
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "hop-decision-server",
        "version": "1.0.0",
        "engine": "cactus-needle-3",
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "hop-latest", "object": "model", "owned_by": "hop"},
            {"id": "hop-1.0.0", "object": "model", "owned_by": "hop"},
            {"id": "hop-preview", "object": "model", "owned_by": "hop"},
            {"id": "jev-latest", "object": "model", "owned_by": "typesafe-compatibility"},
            {"id": "jev-1.13.0", "object": "model", "owned_by": "typesafe-compatibility"},
        ],
    }


@app.post("/v1/systemone", response_model=SystemOneResponse)
async def system_one_endpoint(
    req: SystemOneRequest,
    authorization: Optional[str] = Header(None),
):
    expected_key = os.environ.get("HOP_API_KEY")
    if expected_key:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid Bearer token")
        token = authorization.split("Bearer ", 1)[1].strip()
        if token != expected_key:
            raise HTTPException(status_code=401, detail="Unauthorized: invalid API key")

    try:
        engine = HopEngine.get_instance()
        response = engine.evaluate(
            state=req.state,
            questions=req.questions,
            model=req.model,
        )
        return response
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))
