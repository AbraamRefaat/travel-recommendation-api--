"""
Interest-Based POI Search Module
=================================
PIPELINE:
  1. At startup: Load Cairo_Giza_POI_Database_v3.xlsx, build a combined text
     string per POI, encode all rows with Sentence Transformer (all-MiniLM-L6-v2),
     and cache the numpy embedding matrix in memory.
  2. search_by_interest(user_query, top_k):
     - Encode the user's free-text query with the same model.
     - Compute cosine similarity between the query vector and all cached POI vectors.
     - Return the top_k rows as a list of dicts.
  3. get_gemini_recommendation(user_query, top_pois):
     - Build a structured prompt with the matched POIs.
     - Call Gemini 1.5-flash via google-generativeai.
     - Return the friendly recommendation text.
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

# ---------------------------------------------------------------------------
# STEP 1  —  Load & embed at startup
# ---------------------------------------------------------------------------

def init_interest_search(excel_path: str = "Cairo_Giza_POI_Database_v3.xlsx") -> None:
    """
    Load the POI Excel file, build combined text strings, encode them with the
    Sentence Transformer model, and cache everything in module-level variables.

    Call this ONCE during application startup — not on every request.
    """
    global _poi_df, _poi_vectors, _st_model

    print("🔍 [InterestSearch] Loading POI database…")
    df = pd.read_excel(excel_path, sheet_name="Cairo & Giza POIs")
    _poi_df = df

    # Build the combined text used for embedding
    def build_text(row) -> str:
        name      = row.get("Name", "")
        category  = row.get("Category", "")
        sub       = row.get("Sub-category", "")
        in_out    = row.get("Indoor / outdoor", "")
        return f"{name}. Category: {category}. Sub-category: {sub}. Indoor/Outdoor: {in_out}."

    texts = df.apply(build_text, axis=1).tolist()

    print("🔍 [InterestSearch] Loading Sentence Transformer model (all-MiniLM-L6-v2)…")
    _st_model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"🔍 [InterestSearch] Encoding {len(texts)} POIs…")
    _poi_vectors = _st_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    print(f"✅ [InterestSearch] Ready — {len(texts)} POI embeddings cached.")


# ---------------------------------------------------------------------------
# STEP 2  —  Cosine-similarity search
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

    # Encode the query
    query_vector = _st_model.encode([user_query], convert_to_numpy=True)  # shape (1, 384)

    # Cosine similarity → shape (1, N)
    sims = cosine_similarity(query_vector, _poi_vectors)[0]

    # Get indices of top-k highest similarities
    top_indices = np.argsort(sims)[::-1][:top_k]

    results = []
    for idx in top_indices:
        row = _poi_df.iloc[idx].to_dict()
        # Convert numpy / pandas types to plain Python types for JSON serialisation
        clean_row = {}
        for k, v in row.items():
            if isinstance(v, (np.integer,)):
                clean_row[k] = int(v)
            elif isinstance(v, (np.floating,)):
                clean_row[k] = float(v)
            elif pd.isna(v) if not isinstance(v, (list, dict, np.ndarray)) else False:
                clean_row[k] = None
            else:
                clean_row[k] = v
        results.append(clean_row)

    return results


# ---------------------------------------------------------------------------
# STEP 3  —  Gemini LLM recommendation
# ---------------------------------------------------------------------------

def get_gemini_recommendation(user_query: str, top_pois: list[dict]) -> str:
    """
    Send the matched POIs and the user's original query to Gemini 1.5-flash.

    Returns
    -------
    str  — the friendly recommendation text from Gemini.

    Raises
    ------
    EnvironmentError  — if GEMINI_API_KEY is not set.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable is not set. "
            "Please add it to your environment or .env file."
        )

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Build the POI details block for the prompt
    poi_lines = []
    for i, poi in enumerate(top_pois, start=1):
        poi_lines.append(
            f"{i}. {poi.get('Name', 'N/A')}\n"
            f"   Category: {poi.get('Category', 'N/A')} — {poi.get('Sub-category', 'N/A')}\n"
            f"   Opening Hours: {poi.get('Opening hours', 'N/A')}\n"
            f"   Entry Cost: {poi.get('Entry cost (EGP)', 'N/A')} EGP\n"
            f"   Estimated Visit Duration: {poi.get('Estimated visit duration', 'N/A')}\n"
            f"   Indoor/Outdoor: {poi.get('Indoor / outdoor', 'N/A')}"
        )
    poi_block = "\n\n".join(poi_lines)

    prompt = (
        "You are a helpful Cairo & Giza travel assistant.\n\n"
        f'The user is interested in: "{user_query}"\n\n'
        "Based on their interest, here are the most relevant places found in our database:\n\n"
        f"{poi_block}\n\n"
        "Please give the user a friendly, personalized recommendation explaining why each place "
        "matches their interest, and include any helpful visit tips."
    )

    response = model.generate_content(prompt)
    return response.text
