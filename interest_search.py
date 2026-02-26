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
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Load environment variables from .env file
load_dotenv()

# ---------------------------------------------------------------------------
# Module-level cache — populated once at startup
# ---------------------------------------------------------------------------
_qdrant_client: QdrantClient = None
_collection_name: str = None
_st_model: SentenceTransformer = None
_gemini_client = None          # google.genai.Client instance


# ---------------------------------------------------------------------------
# STEP 1  —  Load & embed at startup
# ---------------------------------------------------------------------------

def init_interest_search() -> None:
    """
    Initialise Qdrant client, load Sentence Transformer, and Gemini client.
    """
    global _qdrant_client, _collection_name, _st_model, _gemini_client

    print("🔍 [InterestSearch] Initialising Qdrant client…")
    host = os.environ.get("QDRANT_HOST", "localhost")
    port = int(os.environ.get("QDRANT_PORT", 6333))
    _collection_name = os.environ.get("QDRANT_COLLECTION", "pois")
    print(f"🔍 [InterestSearch] Connecting to Qdrant at {host}:{port}...")
    _qdrant_client = QdrantClient(host=host, port=port, https=(port == 443))

    print("🔍 [InterestSearch] Loading Sentence Transformer…")
    _st_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Gemini client — created once, reused on every request
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        # We removed the forced 'v1' api_version to allow the SDK to select the best one for each model.
        _gemini_client = google_genai.Client(api_key=api_key)
        print("✅ [InterestSearch] Gemini client ready (google-genai SDK).")
    else:
        print("⚠️  [InterestSearch] GEMINI_API_KEY not set — Gemini disabled.")

    print(f"✅ [InterestSearch] Ready — Connected to Qdrant collection '{_collection_name}'.")

def get_st_model() -> SentenceTransformer:
    return _st_model

def get_qdrant_client() -> QdrantClient:
    return _qdrant_client


# ---------------------------------------------------------------------------
# STEP 2  —  Cosine-similarity search
# ---------------------------------------------------------------------------

def extract_interests(user_query: str) -> list[str]:
    """
    Use Gemini to extract a list of distinct interests from a user query.
    Example: "visit places on Nile and eat Koshary" -> ["visit places on Nile", "eat Koshary"]
    """
    if _gemini_client is None:
        # Fallback if Gemini is not available
        return [user_query]

    prompt = (
        f"Analyze the following traveler's request: \"{user_query}\"\n\n"
        "Your Task:\n"
        "1. Extract distinct travel interests or activities from the request.\n"
        "2. Return ONLY a JSON object with a list of interests like this:\n"
        "{\"interests\": [\"interest 1\", \"interest 2\"]}\n"
        "3. If there is only one interest, return it in the list.\n"
        "Do not include any other text or reasoning."
    )

    _MODELS_TO_TRY = ["gemini-3-flash-preview"]
    
    for model_name in _MODELS_TO_TRY:
        try:
            print(f"📡 [InterestSearch] Extracting interests with model '{model_name}'...")
            response = _gemini_client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            text = response.text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            
            import json
            data = json.loads(text)
            interests = data.get("interests", [])
            if interests:
                print(f"✅ [InterestSearch] Extracted interests: {interests}")
                return interests
        except Exception as e:
            print(f"⚠️  [InterestSearch] Interest extraction failed for '{model_name}': {e}")
            continue
    
    return [user_query]

def search_by_interest(user_query: str, top_k: int = 5) -> list[dict]:
    if _qdrant_client is None or _st_model is None:
        raise RuntimeError(
            "[InterestSearch] Not initialised yet. Ensure Qdrant is running."
        )

    # 1. Extract multiple interests if present
    interests = extract_interests(user_query)
    
    all_results = []
    seen_ids = set()

    # 2. Search for each interest separately
    for interest in interests:
        query_vector = _st_model.encode(interest).tolist()
        try:
            search_result = _qdrant_client.query_points(
                collection_name=_collection_name,
                query=query_vector,
                limit=top_k
            ).points
        except AttributeError:
            search_result = _qdrant_client.search(
                collection_name=_collection_name,
                query_vector=query_vector,
                limit=top_k
            )
        
        for hit in search_result:
            poi_id = hit.id
            if poi_id not in seen_ids:
                poi = hit.payload
                poi['id'] = poi_id
                # Track which interest this result was matched for (optional, for debugging)
                poi['_matched_for'] = interest 
                all_results.append(poi)
                seen_ids.add(poi_id)

    # 3. If we have multiple interests, we might have too many results.
    # We should ensure we have at least 5 but maybe limit the total to something reasonable like 10
    # to avoid overwhelming Gemini, while keeping variety.
    return all_results[:10] if len(interests) > 1 else all_results[:top_k]


# ---------------------------------------------------------------------------
# STEP 3  —  Gemini recommendation (new google-genai SDK)
# ---------------------------------------------------------------------------

def get_gemini_recommendation(user_query: str, top_pois: list[dict]) -> list[int]:
    if _gemini_client is None:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set or Gemini client was not initialised."
        )

    # Compact POI block with ID
    lines = []
    for i, p in enumerate(top_pois, 1):
        # Prefer the 'ID' column if it exists, otherwise use index-based id
        pid = p.get('ID') or p.get('id')
        lines.append(
            f"ID: {pid} | {p.get('Name', 'N/A')} | "
            f"{p.get('Category', '')} / {p.get('Sub-category', '')} | "
            f"{p.get('Indoor / outdoor', '')}"
        )

    # Convert lines list to a block for the prompt
    lines_block = "\n".join(lines)

    prompt = (
        f"Analyze the traveler's interest: \"{user_query}\"\n\n"
        f"Here are top-matching potential places with their IDs:\n{lines_block}\n\n"
        "Your Task:\n"
        "1. Select the BEST places (max 3) from the list that collectively cover the traveler's interests.\n"
        "2. If the user has multiple interests (e.g., 'places on Nile' AND 'eat Koshary'), ensure you select at least one place for each interest part.\n"
        "3. Return ONLY the IDs of these places in the following strict JSON format:\n"
        "{\"best_ids\": [ID1, ID2, ...]}\n"
        "Do not include any other text, reasoning, or markdown formatting outside the JSON."
    )

    # Try models in order — first available one wins
    _MODELS_TO_TRY = [
        "gemini-3-flash-preview"
    ]

    last_error = None
    for model_name in _MODELS_TO_TRY:
        try:
            print(f"📡 [InterestSearch] Prompting Gemini with model '{model_name}'...")
            response = _gemini_client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            text = response.text.strip()
            # Clean possible markdown wrap
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            
            import json
            data = json.loads(text)
            # Return up to 3 best IDs
            return data.get("best_ids", [])[:3]
        except Exception as e:
            print(f"⚠️  [InterestSearch] Model '{model_name}' failed: {e}")
            last_error = e
            continue

    raise last_error
