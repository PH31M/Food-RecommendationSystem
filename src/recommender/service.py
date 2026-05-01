from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


@dataclass
class RecommenderArtifacts:
    train_interactions: pd.DataFrame
    user_features: pd.DataFrame
    popularity_features: pd.DataFrame
    recipe_similarity_topk: pd.DataFrame
    recipe_features: pd.DataFrame
    user_factors: np.ndarray
    item_factors: np.ndarray
    user_means: dict[int, float]
    user_to_index: dict[int, int]
    recipe_ids_array: np.ndarray
    recipe_name_map: dict[int, str]
    cf_available: bool
    cf_status: str


class RecommendationService:
    def __init__(self, repo_root: Path | None = None) -> None:
        self.repo_root = repo_root or Path(__file__).resolve().parents[2]
        self.artifacts_dir = self.repo_root / "artifacts"
        self._cache: RecommenderArtifacts | None = None

    def _load_artifacts(self) -> RecommenderArtifacts:
        if self._cache is not None:
            return self._cache

        features_dir = self.artifacts_dir / "features"
        models_dir = self.artifacts_dir / "models"

        train_interactions = pd.read_parquet(features_dir / "train_interactions.parquet")
        user_features = pd.read_parquet(features_dir / "user_features.parquet")
        popularity_features = pd.read_parquet(features_dir / "popularity_features.parquet")
        recipe_similarity_topk = pd.read_parquet(features_dir / "recipe_similarity_topk.parquet")
        recipe_features = pd.read_parquet(features_dir / "recipe_features.parquet")
        recipe_index_map = pd.read_parquet(features_dir / "recipe_index_map.parquet")

        svd_artifact = joblib.load(models_dir / "svd_model.joblib")
        user_factors = svd_artifact["user_factors"]
        item_factors = svd_artifact["item_factors"]
        user_means = {int(k): float(v) for k, v in svd_artifact["user_means"].items()}

        user_index_df = pd.read_csv(models_dir / "user_index_map.csv")
        user_to_index = {
            int(row.user_id): int(row.matrix_index) for row in user_index_df.itertuples(index=False)
        }

        recipes_path = self.repo_root / "data" / "processed" / "recipes_clean.csv"
        recipes_df = pd.read_csv(recipes_path, usecols=["id", "name"])
        recipe_name_map = {
            int(row.id): str(row.name) for row in recipes_df.dropna(subset=["id"]).itertuples(index=False)
        }

        # Align CF item id space to SVD factors.
        # SVD item factors come from model artifact and may be out-of-sync with
        # current feature artifacts if notebooks were rerun partially.
        n_cf_items = int(item_factors.shape[1])
        item_ids_from_map = (
            recipe_index_map.sort_values("matrix_index")["recipe_id"].to_numpy(dtype=np.int64)
        )
        if len(item_ids_from_map) >= n_cf_items:
            recipe_ids_array = item_ids_from_map[:n_cf_items]
            if len(item_ids_from_map) == n_cf_items:
                cf_available = True
                cf_status = "ok"
            else:
                cf_available = True
                cf_status = (
                    "mismatch_detected: recipe_index_map has more items than SVD; "
                    "truncated to SVD size"
                )
        else:
            # Cannot map all SVD columns to recipe ids reliably.
            recipe_ids_array = item_ids_from_map
            cf_available = False
            cf_status = (
                "unavailable: recipe_index_map has fewer items than SVD; "
                "please retrain notebook 04 to realign artifacts"
            )
        popularity_features = popularity_features.sort_values(
            "popularity_score", ascending=False
        ).reset_index(drop=True)

        self._cache = RecommenderArtifacts(
            train_interactions=train_interactions,
            user_features=user_features,
            popularity_features=popularity_features,
            recipe_similarity_topk=recipe_similarity_topk,
            recipe_features=recipe_features,
            user_factors=user_factors,
            item_factors=item_factors,
            user_means=user_means,
            user_to_index=user_to_index,
            recipe_ids_array=recipe_ids_array,
            recipe_name_map=recipe_name_map,
            cf_available=cf_available,
            cf_status=cf_status,
        )
        return self._cache

    def health(self) -> dict[str, Any]:
        a = self._load_artifacts()
        return {
            "status": "ok",
            "train_interactions": int(len(a.train_interactions)),
            "users": int(a.user_features["user_id"].nunique()),
            "recipes": int(len(a.recipe_ids_array)),
            "cf_available": a.cf_available,
            "cf_status": a.cf_status,
        }

    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        max_calories: float | None = None,
        max_minutes: float | None = None,
        model: str = "auto",
    ) -> list[dict[str, Any]]:
        a = self._load_artifacts()
        top_k = max(1, min(int(top_k), 100))
        model = model.lower().strip()
        allowed_models = {"auto", "popularity", "content", "cf", "hybrid"}
        if model not in allowed_models:
            raise ValueError(f"Unsupported model '{model}'. Supported: {sorted(allowed_models)}")

        seen = set(
            a.train_interactions.loc[a.train_interactions["user_id"] == user_id, "recipe_id"].tolist()
        )

        user_row = a.user_features.loc[a.user_features["user_id"] == user_id]
        is_cold_user = user_row.empty

        if model == "popularity":
            return self._recommend_popularity(a, top_k, seen, max_calories, max_minutes)
        if model == "content":
            content_scores = self._content_scores(a, user_id, seen)
            ranked = sorted(content_scores.items(), key=lambda x: x[1], reverse=True)
            return self._to_response(a, ranked, "content", top_k, max_calories, max_minutes)
        if model == "cf":
            cf_scores = self._cf_scores(a, user_id, seen)
            ranked = sorted(cf_scores.items(), key=lambda x: x[1], reverse=True)
            return self._to_response(a, ranked, "cf", top_k, max_calories, max_minutes)
        if model == "hybrid":
            if is_cold_user:
                return self._recommend_popularity(a, top_k, seen, max_calories, max_minutes)
            rating_count = int(user_row["rating_count"].iloc[0])
            alpha = 0.7 if rating_count < 5 else 0.5
            return self._recommend_hybrid(
                a, user_id, seen, top_k, max_calories, max_minutes, alpha=alpha
            )

        # model == "auto"
        if is_cold_user:
            return self._recommend_popularity(a, top_k, seen, max_calories, max_minutes)

        rating_count = int(user_row["rating_count"].iloc[0])
        alpha = 0.7 if rating_count < 5 else 0.5
        return self._recommend_hybrid(a, user_id, seen, top_k, max_calories, max_minutes, alpha=alpha)

    def _recommend_hybrid(
        self,
        a: RecommenderArtifacts,
        user_id: int,
        seen: set[int],
        top_k: int,
        max_calories: float | None,
        max_minutes: float | None,
        alpha: float,
    ) -> list[dict[str, Any]]:
        content_scores = self._content_scores(a, user_id, seen)
        cf_scores = self._cf_scores(a, user_id, seen)

        if not content_scores and not cf_scores:
            return self._recommend_popularity(a, top_k, seen, max_calories, max_minutes)
        if not content_scores:
            ranked = sorted(cf_scores.items(), key=lambda x: x[1], reverse=True)
            return self._to_response(a, ranked, "cf", top_k, max_calories, max_minutes)
        if not cf_scores:
            ranked = sorted(content_scores.items(), key=lambda x: x[1], reverse=True)
            return self._to_response(a, ranked, "content", top_k, max_calories, max_minutes)

        merged_ids = list(set(content_scores.keys()) | set(cf_scores.keys()))
        content_arr = np.array([content_scores.get(rid, np.nan) for rid in merged_ids], dtype=float)
        cf_arr = np.array([cf_scores.get(rid, np.nan) for rid in merged_ids], dtype=float)

        c_med = np.nanmedian(content_arr) if np.isnan(content_arr).any() else 0.0
        f_med = np.nanmedian(cf_arr) if np.isnan(cf_arr).any() else 0.0
        content_arr = np.where(np.isnan(content_arr), c_med, content_arr)
        cf_arr = np.where(np.isnan(cf_arr), f_med, cf_arr)

        content_norm = self._min_max(content_arr)
        cf_norm = self._min_max(cf_arr)
        hybrid = alpha * content_norm + (1.0 - alpha) * cf_norm

        ranked = sorted(zip(merged_ids, hybrid.tolist()), key=lambda x: x[1], reverse=True)
        return self._to_response(a, ranked, "hybrid", top_k, max_calories, max_minutes)

    @staticmethod
    def _min_max(values: np.ndarray) -> np.ndarray:
        min_v = np.min(values)
        max_v = np.max(values)
        if np.isclose(min_v, max_v):
            return np.full_like(values, 0.5, dtype=float)
        return (values - min_v) / (max_v - min_v)

    def _content_scores(
        self, a: RecommenderArtifacts, user_id: int, seen: set[int]
    ) -> dict[int, float]:
        user_hist = a.train_interactions[
            (a.train_interactions["user_id"] == user_id) & (a.train_interactions["rating"] >= 4)
        ]["recipe_id"].tolist()
        if not user_hist:
            return {}

        score_bag: dict[int, list[float]] = {}
        for recipe_id in user_hist:
            neighbors = a.recipe_similarity_topk[a.recipe_similarity_topk["recipe_id"] == recipe_id]
            if neighbors.empty:
                continue
            for row in neighbors.itertuples(index=False):
                rid = int(row.neighbor_recipe_id)
                if rid in seen:
                    continue
                score_bag.setdefault(rid, []).append(float(row.similarity))
        return {rid: float(np.mean(vals)) for rid, vals in score_bag.items()}

    def _cf_scores(self, a: RecommenderArtifacts, user_id: int, seen: set[int]) -> dict[int, float]:
        if not a.cf_available:
            return {}
        if user_id not in a.user_to_index:
            return {}

        u_idx = a.user_to_index[user_id]
        user_vector = a.user_factors[u_idx]
        scores = user_vector @ a.item_factors
        scores = scores + a.user_means.get(user_id, 0.0)

        n_items = len(scores)
        if seen:
            mask = ~np.isin(a.recipe_ids_array, list(seen))
            candidate_idx = np.where(mask)[0]
        else:
            candidate_idx = np.arange(len(a.recipe_ids_array))

        # Defensive guard in case stale artifacts cause any index mismatch.
        candidate_idx = candidate_idx[candidate_idx < n_items]
        if len(candidate_idx) == 0:
            return {}

        candidate_scores = scores[candidate_idx]
        take_n = min(2000, len(candidate_idx))
        top_local = np.argpartition(-candidate_scores, take_n - 1)[:take_n]
        top_local = top_local[np.argsort(-candidate_scores[top_local])]
        indices = candidate_idx[top_local]

        return {
            int(a.recipe_ids_array[idx]): float(candidate_scores[local_idx])
            for local_idx, idx in enumerate(indices)
        }

    def _recommend_popularity(
        self,
        a: RecommenderArtifacts,
        top_k: int,
        seen: set[int],
        max_calories: float | None,
        max_minutes: float | None,
    ) -> list[dict[str, Any]]:
        df = a.popularity_features
        if seen:
            df = df[~df["recipe_id"].isin(seen)]
        ranked = list(zip(df["recipe_id"].astype(int).tolist(), df["popularity_score"].astype(float).tolist()))
        return self._to_response(a, ranked, "popularity", top_k, max_calories, max_minutes)

    def _to_response(
        self,
        a: RecommenderArtifacts,
        ranked: list[tuple[int, float]],
        source: str,
        top_k: int,
        max_calories: float | None,
        max_minutes: float | None,
    ) -> list[dict[str, Any]]:
        if not ranked:
            return []

        df = pd.DataFrame(ranked, columns=["recipe_id", "score"])
        features = a.recipe_features[["recipe_id", "calories", "minutes"]]
        df = df.merge(features, on="recipe_id", how="left")

        if max_calories is not None:
            df = df[df["calories"] <= max_calories]
        if max_minutes is not None:
            df = df[df["minutes"] <= max_minutes]

        df = df.head(top_k).copy()
        if df.empty:
            return []

        df["name"] = df["recipe_id"].map(a.recipe_name_map).fillna("unknown_recipe")
        df["source"] = source
        df["rank"] = range(1, len(df) + 1)

        return [
            {
                "rank": int(row.rank),
                "recipe_id": int(row.recipe_id),
                "name": str(row.name),
                "score": float(row.score),
                "source": str(row.source),
                "calories": None if pd.isna(row.calories) else float(row.calories),
                "minutes": None if pd.isna(row.minutes) else float(row.minutes),
            }
            for row in df.itertuples(index=False)
        ]

