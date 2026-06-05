"""
Central, env-driven configuration for the recommendation engine.

Every tunable number lives here instead of being hardcoded across the codebase.
All values can be overridden with environment variables (e.g. in a .env file or
the deployment dashboard) without touching the source — that is what makes the
engine "dynamic". Sensible fallbacks are provided so it works out-of-the-box.
"""
import os
from dataclasses import dataclass, field
from typing import Dict


def _f(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _i(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return int(default)


@dataclass(frozen=True)
class EngineConfig:
    # ---- Data parsing fallbacks ----
    default_visit_hours: float = field(default_factory=lambda: _f("DEFAULT_VISIT_HOURS", 1.5))
    default_day_start: float = field(default_factory=lambda: _f("DEFAULT_DAY_START", 9.0))
    default_day_end: float = field(default_factory=lambda: _f("DEFAULT_DAY_END", 21.0))
    qdrant_scroll_limit: int = field(default_factory=lambda: _i("QDRANT_SCROLL_LIMIT", 1000))

    # ---- Travel model ----
    travel_speed_kmh: float = field(default_factory=lambda: _f("TRAVEL_SPEED_KMH", 20.0))
    min_travel_hours: float = field(default_factory=lambda: _f("MIN_TRAVEL_HOURS", 0.25))
    max_travel_hours: float = field(default_factory=lambda: _f("MAX_TRAVEL_HOURS", 1.5))

    # ---- Ranking weights ----
    w_semantic: float = field(default_factory=lambda: _f("RANK_W_SEMANTIC", 10.0))
    w_interest_category: float = field(default_factory=lambda: _f("RANK_W_INTEREST_CATEGORY", 10.0))
    w_interest_description: float = field(default_factory=lambda: _f("RANK_W_INTEREST_DESCRIPTION", 2.0))
    base_score_no_match: float = field(default_factory=lambda: _f("RANK_BASE_NO_MATCH", 1.0))

    # ---- Pace model (fixed stops per day, overridable via env) ----
    max_items_relaxed: int = field(default_factory=lambda: _i("MAX_ITEMS_RELAXED", 2))
    max_items_moderate: int = field(default_factory=lambda: _i("MAX_ITEMS_MODERATE", 3))
    max_items_packed: int = field(default_factory=lambda: _i("MAX_ITEMS_PACKED", 5))

    # ---- Optimizer penalties ----
    diversity_penalty: float = field(default_factory=lambda: _f("DIVERSITY_PENALTY", 5.0))
    distance_penalty_per_km: float = field(default_factory=lambda: _f("DISTANCE_PENALTY_PER_KM", 0.5))

    # ---- Price-tier suitability bonuses ----
    price_match_bonus: float = field(default_factory=lambda: _f("PRICE_MATCH_BONUS", 3.0))
    price_tolerate_bonus: float = field(default_factory=lambda: _f("PRICE_TOLERATE_BONUS", 1.0))
    price_mismatch_penalty: float = field(default_factory=lambda: _f("PRICE_MISMATCH_PENALTY", 3.0))

    # ---- AI candidate generation ----
    search_depth_multiplier: int = field(default_factory=lambda: _i("SEARCH_DEPTH_MULTIPLIER", 6))
    rerank_pool_multiplier: int = field(default_factory=lambda: _i("RERANK_POOL_MULTIPLIER", 4))
    keyword_boost_per_interest: float = field(default_factory=lambda: _f("KEYWORD_BOOST_PER_INTEREST", 0.1))
    interest_primary_threshold: float = field(default_factory=lambda: _f("INTEREST_PRIMARY_THRESHOLD", 0.9))
    interest_secondary_threshold: float = field(default_factory=lambda: _f("INTEREST_SECONDARY_THRESHOLD", 0.6))
    fallback_query: str = field(default_factory=lambda: os.environ.get(
        "FALLBACK_QUERY", "Popular tourist attractions and points of interest"))

    @property
    def max_items(self) -> Dict[str, int]:
        return {
            "relaxed": self.max_items_relaxed,
            "moderate": self.max_items_moderate,
            "packed": self.max_items_packed,
        }

CONFIG = EngineConfig()
