"""
Interest-Based POI Search Module
=================================
Uses the new `google-genai` SDK (>=1.0.0) which replaces the deprecated
`google-generativeai` package and supports gemini-1.5-flash via the v1 API.
"""

import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai as google_genai

# ---------------------------------------------------------------------------
# Module-level cache — populated once at startup
# ---------------------------------------------------------------------------
_poi_df: pd.DataFrame = None
_poi_vectors: np.ndarray = None
_st_model: SentenceTransformer = None
_gemini_client = None          # google.genai.Client instance


# ---------------------------------------------------------------------------
# STEP 1  —  Load & embed at startup
# ---------------------------------------------------------------------------

def init_interest_search(excel_path: str = "Cairo_Giza_POI_Database_v3.xlsx") -> None:
    """
    Load the POI Excel file, encode all rows with Sentence Transformer,
    and cache everything. Also creates the Gemini client ONCE for speed.
    """
    global _poi_df, _poi_vectors, _st_model, _gemini_client

    print("🔍 [InterestSearch] Loading POI database…")
    df = pd.read_excel(excel_path, sheet_name="Cairo & Giza POIs")
    _poi_df = df

    def build_text(row) -> str:
        return (
            f"{row.get('Name', '')}. "
            f"Category: {row.get('Category', '')}. "
            f"Sub-category: {row.get('Sub-category', '')}. "
            f"Indoor/Outdoor: {row.get('Indoor / outdoor', '')}."
        )

    texts = df.apply(build_text, axis=1).tolist()

    print("🔍 [InterestSearch] Loading Sentence Transformer…")
    _st_model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"🔍 [InterestSearch] Encoding {len(texts)} POIs…")
    _poi_vectors = _st_model.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Gemini client — created once, reused on every request
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        _gemini_client = google_genai.Client(api_key=api_key)
        print("✅ [InterestSearch] Gemini client ready (google-genai SDK).")
    else:
        print("⚠️  [InterestSearch] GEMINI_API_KEY not set — Gemini disabled.")

    print(f"✅ [InterestSearch] Ready — {len(texts)} POI embeddings cached.")


# ---------------------------------------------------------------------------
# STEP 2  —  Cosine-similarity search
# ---------------------------------------------------------------------------

def search_by_interest(user_query: str, top_k: int = 5) -> list[dict]:
    if _poi_df is None or _poi_vectors is None or _st_model is None:
        raise RuntimeError(
            "[InterestSearch] Not initialised yet. Retry in ~60 seconds."
        )

    query_vector = _st_model.encode(
        [user_query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # Fast dot product (vectors already normalised)
    sims = (query_vector @ _poi_vectors.T)[0]
    top_indices = np.argpartition(sims, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]

    results = []
    for idx in top_indices:
        row = _poi_df.iloc[idx].to_dict()
        clean = {}
        for k, v in row.items():
            if isinstance(v, np.integer):
                clean[k] = int(v)
            elif isinstance(v, np.floating):
                clean[k] = float(v)
            elif not isinstance(v, (list, dict, np.ndarray)) and pd.isna(v):
                clean[k] = None
            else:
                clean[k] = v
        results.append(clean)

    return results


# ---------------------------------------------------------------------------
# STEP 3  —  Gemini recommendation (new google-genai SDK)
# ---------------------------------------------------------------------------

def get_gemini_recommendation(user_query: str, top_pois: list[dict]) -> str:
    if _gemini_client is None:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set or Gemini client was not initialised."
        )

    # Compact POI block
    lines = []
    for i, p in enumerate(top_pois, 1):
        lines.append(
            f"{i}. {p.get('Name', 'N/A')} | "
            f"{p.get('Category', '')} / {p.get('Sub-category', '')} | "
            f"Cost: {p.get('Entry cost (EGP)', 'N/A')} EGP | "
            f"Hours: {p.get('Opening hours', 'N/A')} | "
            f"{p.get('Indoor / outdoor', '')}"
        )

    prompt = (
        f'User interest: "{user_query}"\n\n'
        f"Top matching Cairo & Giza places:\n" + "\n".join(lines) + "\n\n"
        "Give a short, friendly recommendation (2-3 sentences per place) "
        "explaining why each matches the interest and one helpful visit tip. "
        "Be concise."
    )

    response = _gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    return response.text
