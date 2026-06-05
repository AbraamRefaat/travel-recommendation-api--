import os
import uuid
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Load environment variables
load_dotenv()

# Constants
COLUMNS = [
    "ID",
    "Name",
    "Latitude / Longitude",
    "Opening time",
    "Closing time",
    "Category",
    "Sub-category",
    "Estimated visit duration",
    "Indoor / outdoor",
    "Price range",
]

VECTOR_SIZE = 384
ST_MODEL_NAME = "all-MiniLM-L6-v2"
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "pois")
EXCEL_FILE = "Cairo_Giza_POI_Database_v3.xlsx"

def build_sentence(row: dict) -> str:
    name        = row.get("Name", "Unknown")
    cat         = row.get("Category", "")
    sub         = row.get("Sub-category", "")
    indoor      = row.get("Indoor / outdoor", "")
    open_time   = row.get("Opening time", "")
    close_time  = row.get("Closing time", "")
    price_range = row.get("Price range", "")
    duration    = row.get("Estimated visit duration", "")
    location    = row.get("Latitude / Longitude", "")

    # Map price range symbols to readable text
    price_labels = {"$": "Budget", "$$": "Moderate", "$$$": "Luxury"}
    price_text = price_labels.get(str(price_range).strip(), str(price_range).strip())

    parts = [f"{name} is a {cat}"]
    if sub:
        parts[0] += f" ({sub})"
    parts[0] += "."
    if indoor:
        parts.append(f"It is {indoor}.")
    if duration:
        parts.append(f"Estimated visit duration: {duration}.")
    if open_time or close_time:
        parts.append(f"Opening hours: {open_time} - {close_time}.")
    if price_text:
        parts.append(f"Price range: {price_text}.")
    if location:
        parts.append(f"Location: {location}.")
    return " ".join(parts)

def safe_int_id(raw_id) -> int:
    try:
        return int(float(str(raw_id)))
    except (ValueError, TypeError):
        return None

def main():
    print(f"🚀 Starting CLI ingestion...")
    
    # Load model
    print("⚙️ Loading Sentence Transformer model...")
    model = SentenceTransformer(ST_MODEL_NAME)
    
    # Connect to Qdrant
    print(f"📡 Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, https=(QDRANT_PORT == 443))
    
    # Load Excel
    print(f"📂 Reading Excel file: {EXCEL_FILE}")
    df = pd.read_excel(EXCEL_FILE)
    df.columns = [str(c).strip() for c in df.columns]
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[COLUMNS]
    
    rows = df.to_dict(orient="records")
    
    # Create collection if needed
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in existing:
        print(f"   Creating collection '{COLLECTION_NAME}'...")
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
    
    print(f"🚀 Ingesting {len(rows)} rows...")
    for i, row in enumerate(rows, start=1):
        try:
            sentence = build_sentence(row)
            vector = model.encode(sentence).tolist()
            
            raw_id = row.get("ID")
            point_id = safe_int_id(raw_id)
            if point_id is None:
                point_id = str(uuid.uuid4())
            
            payload = {k: str(v) if v is not None else "" for k, v in row.items()}
            
            qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=[PointStruct(id=point_id, vector=vector, payload=payload)],
            )
            if i % 10 == 0:
                print(f"   [{i}/{len(rows)}] Upserted: {row.get('Name')}")
        except Exception as e:
            print(f"   ❌ Error at row {i}: {e}")

    print("✅ Ingestion complete!")

if __name__ == "__main__":
    main()
