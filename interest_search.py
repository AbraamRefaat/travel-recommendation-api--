"""
Interest-Based POI Search Module
=================================
PIPELINE:
  1. At startup: Load Cairo_Giza_POI_Database_v3.xlsx, build a combined text
     string per POI, encode all rows with Sentence Transformer (all-MiniLM-L6-v2),
     and cache the numpy embedding matrix in memory.
  2. search_by_interest(user_query, top_k):
     - Encode the user's free-text query with the same model (~50ms).
     - Compute cosine similarity against cached vectors (~5ms).
     - Return the top_k rows as a list of dicts.
  3. get_gemini_recommendation(user_query, top_pois):
     - Build a compact prompt with the matched POIs.
     - Call Gemini 1.5-flash (configured ONCE at module level for speed).
     - Return the friendly recommendation text.

Speed optimisations applied:
  - Gemini client configured once at init (not on every request).
  - Compact prompt — fewer tokens = faster Gemini response.
  - Sentence Transformer encoding uses batch_size=1, no progress bar.
  - numpy argsort for O(N) top-k selection.
"""

import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# ---------------------------------------------------------------------------
# Module-level cache — populated once when init_interest_search() is called
# ---------------------------------------------------------------------------
_poi_df: pd.DataFrame = None          # Full dataframe of POIs
_poi_vectors: np.ndarray = None       # (N, 384) embedding matrix
_st_model: SentenceTransformer = None # Shared sentence transformer model
_gemini_model = None                  # Gemini model instance (created once)

# ---------------------------------------------------------------------------
# STEP 1  —  Load & embed at startup
# ---------------------------------------------------------------------------

def init_interest_search(excel_path: str = "Cairo_Giza_POI_Database_v3.xlsx") -> None:
    """
    Load the POI Excel file, build combined text strings, encode them with the
    Sentence Transformer model, and cache everything in module-level variables.
    Also initialises the Gemini client ONCE for faster per-request calls.

    Call this ONCE during application startup — not on every request.
    """
    global _poi_df, _poi_vectors, _st_model, _gemini_model

    print("🔍 [InterestSearch] Loading POI database…")
    df = pd.read_excel(excel_path, sheet_name="Cairo & Giza POIs")
    _poi_df = df

    # Build the combined text used for embedding
    def build_text(row) -> str:
        name     = row.get("Name", "")
        category = row.get("Category", "")
        sub      = row.get("Sub-category", "")
        in_out   = row.get("Indoor / outdoor", "")
        return f"{name}. Category: {category}. Sub-category: {sub}. Indoor/Outdoor: {in_out}."

    texts = df.apply(build_text, axis=1).tolist()

    print("🔍 [InterestSearch] Loading Sentence Transformer model (all-MiniLM-L6-v2)…")
    _st_model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"🔍 [InterestSearch] Encoding {len(texts)} POIs…")
    _poi_vectors = _st_model.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,   # pre-normalise → dot product == cosine sim (faster)
    )

    # --- Initialise Gemini once so there's no auth overhead per request ---
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        _gemini_model = genai.GenerativeModel(
            "gemini-1.5-flash",
            generation_config={"temperature": 0.7, "max_output_tokens": 400},
        )
        print("✅ [InterestSearch] Gemini client ready.")
    else:
        print("⚠️  [InterestSearch] GEMINI_API_KEY not set — Gemini disabled.")

    print(f"✅ [InterestSearch] Ready — {len(texts)} POI embeddings cached.")


# ---------------------------------------------------------------------------
# STEP 2  —  Cosine-similarity search  (fast: ~50 ms total)
# ---------------------------------------------------------------------------

def search_by_interest(user_query: str, top_k: int = 5) -> list[dict]:
    """
    Encode *user_query* and return the top_k most similar POIs as dicts.

    Returns
    -------
    list[dict]  — each dict contains all columns from the dataframe.

    Raises
    ------
    RuntimeError  — if init_interest_search() has not been called yet.
    """
    if _poi_df is None or _poi_vectors is None or _st_model is None:
        raise RuntimeError(
            "[InterestSearch] Module not initialised. "
            "Call init_interest_search() at application startup."
        )

    # Encode & normalise the query (normalised vectors → dot product == cosine)
    query_vector = _st_model.encode(
        [user_query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # Fast dot product similarity (vectors already normalised)
    sims = (query_vector @ _poi_vectors.T)[0]

    # Top-k indices
    top_indices = np.argpartition(sims, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]

    results = []
    for idx in top_indices:
        row = _poi_df.iloc[idx].to_dict()
        clean_row = {}
        for k, v in row.items():
            if isinstance(v, np.integer):
                clean_row[k] = int(v)
            elif isinstance(v, np.floating):
                clean_row[k] = float(v)
            elif not isinstance(v, (list, dict, np.ndarray)) and pd.isna(v):
                clean_row[k] = None
            else:
                clean_row[k] = v
        results.append(clean_row)

    return results


# ---------------------------------------------------------------------------
# STEP 3  —  Gemini LLM recommendation  (compact prompt = faster)
# ---------------------------------------------------------------------------

def get_gemini_recommendation(user_query: str, top_pois: list[dict]) -> str:
    """
    Send the matched POIs and the user's query to Gemini 1.5-flash.
    Uses the pre-configured client (_gemini_model) for speed.

    Returns
    -------
    str  — the friendly recommendation text from Gemini.

    Raises
    ------
    EnvironmentError  — if GEMINI_API_KEY was not set at startup.
    """
    if _gemini_model is None:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable is not set. "
            "Please add it to your environment or .env file."
        )

    # Compact POI block — fewer tokens = faster response
    poi_lines = []
    for i, poi in enumerate(top_pois, start=1):
        poi_lines.append(
            f"{i}. {poi.get('Name', 'N/A')} | {poi.get('Category', '')} / {poi.get('Sub-category', '')} | "
            f"Cost: {poi.get('Entry cost (EGP)', 'N/A')} EGP | "
            f"Hours: {poi.get('Opening hours', 'N/A')} | "
            f"{poi.get('Indoor / outdoor', '')}"
        )
    poi_block = "\n".join(poi_lines)

    prompt = (
        f'User interest: "{user_query}"\n\n'
        f"Top matching Cairo & Giza places:\n{poi_block}\n\n"
        "Give a short, friendly recommendation (2-3 sentences per place) explaining "
        "why each matches the interest and one helpful visit tip. Be concise."
    )

    response = _gemini_model.generate_content(prompt)
    return response.text
