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
    start_time: str = "09:00"
    end_time: str = "17:00"
    
    # Optional fields
    budget_daily: Optional[float] = None
    budget_total: Optional[float] = None
    geo_center: Optional[tuple] = None
    geo_radius_km: float = 20.0
    willingness_to_pay_entry: bool = True
    indoor_preference: str = "neutral"

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

def _load_interest_search_bg():
    """Background thread: loads Sentence Transformer + encodes all POIs."""
    global interest_search_ready
    try:
        init_interest_search("Cairo_Giza_POI_Database_v3.xlsx")
        interest_search_ready = True
        print("✅ [InterestSearch] Embeddings ready — /search-by-interest is live!")
    except Exception as ie:
        print(f"⚠️  [InterestSearch] Could not initialise: {ie}")

@app.on_event("startup")
def startup():
    global system
    print("=" * 60)
    print("🚀 Starting Tourist Recommendation API")
    print("=" * 60)
    
    # Get and display local IPs
    local_ips = get_local_ips()
    if local_ips:
        print("\n📱 Connect your phone to the same network and use:")
        for ip in local_ips:
            print(f"   http://{ip}:8000")
        print()
    
    try:
        print("📂 Loading POI database...")
        system = TouristRecommendationSystem("Cairo_Giza_POI_Database_v3.xlsx")
        print("✅ System ready!")
        print("=" * 60)
        
        # Start broadcast thread for discovery
        broadcast_thread = threading.Thread(target=broadcast_server, daemon=True)
        broadcast_thread.start()
        print("📡 Broadcasting server presence for automatic discovery...")
        print("=" * 60)

        # ── New: load embeddings in background so healthcheck passes instantly ──
        interest_thread = threading.Thread(target=_load_interest_search_bg, daemon=True)
        interest_thread.start()
        print("🔍 [InterestSearch] Loading embeddings in background…")
        
    except Exception as e:
        print(f"❌ ERROR: Failed to load system: {e}")
        print("=" * 60)
        raise

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
        "cost": poi.cost,
        "opening_hours": poi.opening_hours,
        "indoor_outdoor": poi.indoor_outdoor,
        "description": getattr(poi, 'description', ''),
        "score": poi.score
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
            budget_daily=request.budget_daily or 1000.0,
            budget_total=request.budget_total or 5000.0,
            duration_days=request.duration_days,
            pace=request.pace,
            start_time=request.start_time,
            end_time=request.end_time,
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
        total_cost = 0.0
        total_pois = 0
        
        for day, events in raw_itinerary.items():
            result[str(day)] = [event_to_dict(e) for e in events]
            for event in events:
                total_cost += event["poi"].cost
                total_pois += 1
        
        response = {
            "success": True,
            "itinerary": result,
            "summary": {
                "total_days": request.duration_days,
                "total_pois": total_pois,
                "total_cost_egp": round(total_cost, 2),
                "daily_budget": user_profile.budget_daily,
                "budget_remaining": round((user_profile.budget_daily * request.duration_days) - total_cost, 2)
            }
        }
        
        print(f"✅ SUCCESS: Generated {total_pois} POIs across {request.duration_days} days")
        print("=" * 60)
        
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
    """Detailed health check"""
    if system is None:
        return {
            "status": "initializing",
            "system_ready": False
        }
    
    return {
        "status": "healthy",
        "system_ready": True,
        "pois_loaded": len(system.loader.pois)
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
