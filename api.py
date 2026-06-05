"""
Simple FastAPI Server for Tourist Recommendation System
ONE ENDPOINT: POST /recommend
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional
import uvicorn
import socket
import threading
import time
import os
import math
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import your recommendation system
from tourist_recommendation_system import TouristRecommendationSystem, UserProfile

# Import the new interest-based POI search module (self-contained, non-breaking)
from interest_search import init_interest_search, search_by_interest, get_gemini_recommendation

# ============================================================================
# 1. CREATE APP
# ============================================================================
app = FastAPI(title="Tourist Recommendation API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global system instance
system = None

# ============================================================================
# 2. REQUEST MODEL (What Flutter sends)
# ============================================================================
class RecommendationRequest(BaseModel):
    """Matches UserProfile structure"""
    interests: Dict[str, float]
    budget_tier: str = "moderate"
    duration_days: int = 1
    pace: str = "moderate"

    # Optional fields
    budget_daily: Optional[float] = None
    geo_center: Optional[tuple] = None
    geo_radius_km: float = 20.0
    willingness_to_pay_entry: bool = True
    indoor_preference: str = "neutral"

    # Optional free-text interest (triggers semantic search + Gemini)
    specific_interest: Optional[str] = None

# ============================================================================
# 3. NETWORK BROADCAST FOR DISCOVERY
# ============================================================================
def get_local_ips():
    """Get all local IP addresses"""
    ips = []
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None):
            ip = info[4][0]
            if ':' not in ip and not ip.startswith('127.'):  # IPv4, not localhost
                if ip not in ips:
                    ips.append(ip)
    except:
        pass
    return ips

def broadcast_server():
    """Broadcast server presence on UDP for discovery"""
    broadcast_port = 37020
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    ips = get_local_ips()
    
    while True:
        try:
            for ip in ips:
                message = f"NILEQUEST_SERVER:{ip}:8000".encode()
                sock.sendto(message, ('<broadcast>', broadcast_port))
            time.sleep(2)  # Broadcast every 2 seconds
        except Exception as e:
            print(f"Broadcast error: {e}")
            time.sleep(5)

# ============================================================================
# 4. STARTUP - Load System Once
# ============================================================================

# Flag so the /search-by-interest endpoint can tell users to wait if embeddings
# are still loading in the background.
interest_search_ready = False

def _initialize_application_bg():
    """Background startup: Load all models once and initialize the system."""
    global system, interest_search_ready
    try:
        # 1. Initialize semantic search + model loading
        print("🔍 [Startup] Initializing Interest Search & Loading Models…")
        init_interest_search()
        
        # 2. Get shared model
        from interest_search import get_st_model
        shared_model = get_st_model()
        
        # 3. Initialize main recommendation system with shared model
        print("📡 [Startup] Connecting to Qdrant & Initializing System…")
        from tourist_recommendation_system import TouristRecommendationSystem
        system = TouristRecommendationSystem("pois", model=shared_model)
        
        interest_search_ready = True
        print("✅ [Startup] Application fully initialized and ready!")
    except Exception as e:
        print(f"❌ [Startup] Initialization failed: {e}")
        import traceback
        traceback.print_exc()

@app.on_event("startup")
def startup():
    print("=" * 60)
    print("🚀 Starting Tourist Recommendation API")
    print("=" * 60)
    
    # 1. Start background initialization immediately
    init_thread = threading.Thread(target=_initialize_application_bg, daemon=True)
    init_thread.start()
    
    # 2. Start broadcast thread
    broadcast_thread = threading.Thread(target=broadcast_server, daemon=True)
    broadcast_thread.start()
    
    print("📡 Application starting in non-blocking mode.")
    print("📡 Health check /health is live.")
    print("=" * 60)

# ============================================================================
# 5. HELPER - Convert POI object to dict
# ============================================================================
def poi_to_dict(poi):
    """Convert POI dataclass to dictionary for JSON"""
    return {
        "id": poi.id,
        "name": poi.name,
        "lat": poi.lat,
        "lon": poi.lon,
        "category": poi.category,
        "subcategory": poi.subcategory,
        "duration_hours": poi.duration_hours,
        "price_range": getattr(poi, 'price_range', ''),
        "opening_hours": poi.opening_hours,
        "indoor_outdoor": poi.indoor_outdoor,
        "description": getattr(poi, 'description', ''),
        "score": float(poi.score) if (poi.score is not None and not math.isnan(poi.score)) else 0.0
    }

def event_to_dict(event):
    """Convert event dict (with POI object) to fully serializable dict"""
    return {
        "poi": poi_to_dict(event["poi"]),
        "start_time": event["start_time"],
        "end_time": event["end_time"],
        "travel_time_hours": event["travel_time_hours"],
        "reason": event["reason"]
    }

# ============================================================================
# 5. MAIN ENDPOINT - Generate Itinerary
# ============================================================================
@app.post("/recommend")
def recommend(request: RecommendationRequest):
    """
    Generate personalized itinerary
    
    Example Request:
    {
        "interests": {"History": 0.9, "Food": 0.7},
        "budget_tier": "moderate",
        "duration_days": 3,
        "pace": "moderate"
    }
    """
    if system is None:
        raise HTTPException(status_code=503, detail="System not initialized yet. Wait a moment and retry.")
    
    try:
        print("\n" + "=" * 60)
        print("📥 NEW REQUEST")
        print(f"   Interests: {request.interests}")
        print(f"   Duration: {request.duration_days} days")
        print(f"   Budget: {request.budget_tier}")
        print("=" * 60)
        
        # Create UserProfile from request
        user_profile = UserProfile(
            interests=request.interests,
            budget_tier=request.budget_tier,
            budget_daily=request.budget_daily,  # None → derived from tier in UserProfile
            duration_days=request.duration_days,
            pace=request.pace,
            geo_center=tuple(request.geo_center) if request.geo_center else None,
            geo_radius_km=request.geo_radius_km,
            willingness_to_pay_entry=request.willingness_to_pay_entry,
            indoor_preference=request.indoor_preference
        )
        
        # Generate itinerary
        print("🤖 Generating itinerary...")
        raw_itinerary = system.generate_itinerary(user_profile)
        
        # Convert to JSON-serializable format
        print("📦 Converting to JSON...")
        result = {}
        total_pois = 0

        for day, events in raw_itinerary.items():
            result[str(day)] = [event_to_dict(e) for e in events]
            total_pois += len(events)

        response = {
            "success": True,
            "itinerary": result,
            "summary": {
                "total_days": request.duration_days,
                "total_pois": total_pois,
            }
        }
        
        print(f"✅ SUCCESS: Generated {total_pois} POIs across {request.duration_days} days")
        print("=" * 60)

        # ── Optional Step 6: specific interest → semantic search + Gemini ──
        if request.specific_interest and request.specific_interest.strip():
            print(f"🔍 [InterestSearch] specific_interest: '{request.specific_interest}'")
            try:
                # 1. Get up to 10 candidates (increased to handle multi-interest)
                matched_pois = search_by_interest(
                    request.specific_interest.strip(), top_k=5
                )
                
                if not matched_pois:
                    print("⚠️  [InterestSearch] No POIs found for query")
                    raise Exception("No matching POIs found")
                
                # 2. Get top IDs from Gemini (can return up to 3)
                selected_ids = []
                try:
                    selected_ids = get_gemini_recommendation(
                        request.specific_interest.strip(), matched_pois
                    )
                    print(f"✅ [InterestSearch] Gemini selected IDs: {selected_ids}")
                except Exception as gemini_error:
                    print(f"⚠️  [InterestSearch] Gemini ranking failed: {gemini_error}")
                    # Fallback: use top 3 from semantic search
                    selected_ids = [p['id'] for p in matched_pois[:3]]
                    print(f"🔄 [InterestSearch] Using top semantic search results: {selected_ids}")

                # 3. Fetch POI objects for these IDs
                # Reload loader if it was initialised before data was ingested
                if not system.loader.pois:
                    print("🔄 [InterestSearch] Loader is empty — reloading from Qdrant...")
                    system.loader.load_data()

                # Normalise all keys to str to avoid UUID-object vs string mismatches
                full_poi_map = {str(p.id): p for p in system.loader.pois}
                print(f"🗺️  [InterestSearch] full_poi_map has {len(full_poi_map)} entries")

                injected_pois = []
                for sid in selected_ids:
                    p_obj = full_poi_map.get(str(sid))
                    if p_obj:
                        injected_pois.append(p_obj)
                    else:
                        print(f"⚠️  [InterestSearch] POI with ID {sid} not found in loader")
                
                if injected_pois:
                    print(f"💉 [InterestSearch] Injecting {len(injected_pois)} POIs into itinerary...")
                    
                    # 4. Distribute across days
                    # POI 1 -> Day 1, POI 2 -> Day 2 (if exists)
                    for i, poi in enumerate(injected_pois):
                        day_num = i + 1
                        if day_num > request.duration_days:
                            day_num = 1 # Fallback to day 1 if single day trip
                        
                        # Get current list of POI objects for this day
                        # Note: raw_itinerary[day] contains event dicts with a "poi" object
                        if day_num in raw_itinerary:
                            current_day_pois = [e["poi"] for e in raw_itinerary[day_num]]
                            # Avoid duplicates
                            if any(poi.id == p.id for p in current_day_pois):
                                continue
                            
                            current_day_pois.append(poi)
                            
                            # 5. Re-schedule this day
                            print(f"🕒 [InterestSearch] Re-scheduling Day {day_num}...")
                            # generate_timed_itinerary takes Dict[int, List[POI]]
                            temp_plan = {day_num: current_day_pois}
                            new_day_schedule = system.scheduler.generate_timed_itinerary(temp_plan, user_profile)
                            
                            # Update the itinerary
                            raw_itinerary[day_num] = new_day_schedule[day_num]

                    # 6. Re-calculate full response and summary
                    result = {}
                    total_pois = 0
                    for day, events in raw_itinerary.items():
                        result[str(day)] = [event_to_dict(e) for e in events]
                        total_pois += len(events)

                    response["itinerary"] = result
                    response["summary"] = {
                        "total_days": request.duration_days,
                        "total_pois": total_pois,
                    }
                    response["interest_search"] = {
                        "query": request.specific_interest.strip(),
                        "selected_ids": selected_ids,
                        "status": "Injected into itinerary"
                    }

            except Exception as ise:
                print(f"⚠️  [InterestSearch] Injection failed: {ise}")
                import traceback
                traceback.print_exc()

        return response
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        print("=" * 60)
        raise HTTPException(status_code=500, detail=f"Failed to generate itinerary: {str(e)}")

# ============================================================================
# 6. INTEREST-BASED POI SEARCH  (new — does not touch existing routes)
# ============================================================================
class InterestSearchRequest(BaseModel):
    query: str
    top_k: int = 5


@app.post("/search-by-interest")
def search_by_interest_endpoint(request: InterestSearchRequest):
    """
    Interest-based POI search powered by Sentence Transformers + Gemini.

    Request JSON: { "query": "I love art and ancient history", "top_k": 5 }
    Response JSON: { "places": [...], "recommendation": "..." }
    """
    if not request.query or not request.query.strip():
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Query string must not be empty.")

    if not interest_search_ready:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=503,
            detail="Interest search is still loading (embedding model). Please retry in ~60 seconds."
        )


    try:
        print(f"\n🔍 [InterestSearch] Query: '{request.query}' | top_k={request.top_k}")
        top_pois = search_by_interest(request.query, top_k=request.top_k)
    except RuntimeError as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail=str(e))

    try:
        recommendation = get_gemini_recommendation(request.query, top_pois)
    except EnvironmentError as e:
        return {"places": top_pois, "recommendation": None, "error": str(e)}
    except Exception as e:
        return {"places": top_pois, "recommendation": None, "error": f"Gemini error: {str(e)}"}

    return {"places": top_pois, "recommendation": recommendation}


# ============================================================================
# 7. HEALTH CHECK
# ============================================================================
@app.get("/")
def root():
    """Simple health check"""
    return {
        "status": "running",
        "message": "Tourist Recommendation API",
        "ready": system is not None
    }

@app.get("/health")
def health():
    """Satisfy Railway health check immediately."""
    status = "healthy" if (system and interest_search_ready) else "initializing"
    return {
        "status": status,
        "system_ready": system is not None,
        "interest_search_ready": interest_search_ready,
        "pois_loaded": len(system.loader.pois) if system else 0
    }

# ============================================================================
# 7. RUN SERVER
# ============================================================================
if __name__ == "__main__":
    # Get port from environment (for Railway, Render, etc.) or default to 8000
    PORT = int(os.environ.get("PORT", 8000))
    
    print("\n")
    print("=" * 60)
    print("  TOURIST RECOMMENDATION API SERVER")
    print("=" * 60)
    print(f"  Server will start at: http://0.0.0.0:{PORT}")
    print(f"  API Docs: http://0.0.0.0:{PORT}/docs")
    print(f"  Health: http://0.0.0.0:{PORT}/health")
    print("=" * 60)
    print("\n")
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=PORT,
        reload=False  # Disable reload in production
    )
