from __future__ import annotations

from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.recommender.service import RecommendationService


class RecommendRequest(BaseModel):
    user_id: int = Field(..., description="User id for personalized recommendation")
    top_k: int = Field(default=10, ge=1, le=100)
    max_calories: float | None = Field(default=None, gt=0)
    max_minutes: float | None = Field(default=None, gt=0)
    model: Literal["auto", "popularity", "content", "cf", "hybrid"] = "auto"


class RecommendItem(BaseModel):
    rank: int
    recipe_id: int
    name: str
    score: float
    source: str
    calories: float | None = None
    minutes: float | None = None


class RecommendResponse(BaseModel):
    user_id: int
    top_k: int
    items: list[RecommendItem]


app = FastAPI(title="Food Recommendation API", version="1.0.0")
service = RecommendationService()


@app.get("/health")
def health() -> dict:
    return service.health()


@app.post("/recommend", response_model=RecommendResponse)
def recommend(payload: RecommendRequest) -> RecommendResponse:
    items = service.recommend(
        user_id=payload.user_id,
        top_k=payload.top_k,
        max_calories=payload.max_calories,
        max_minutes=payload.max_minutes,
        model=payload.model,
    )
    return RecommendResponse(user_id=payload.user_id, top_k=payload.top_k, items=items)

