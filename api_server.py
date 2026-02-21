from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Tuple, Optional, List
import os
import sys

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tourist_recommendation_system import TouristRecommendationSystem, UserProfile

app = FastAPI(title="Nile Quest Recommendation API", version="1.0.0")

# Enable CORS for Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Flutter app's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Request / Response Schemas ----------

class UserProfileRequest(BaseModel):
    interests: Dict[str, float]
    budget_tier: str = "moderate"   # "budget" | "moderate" | "luxury"
    duration_days: int = 1
    pace: str = "moderate"          # "relaxed" | "moderate" | "packed"
    start_time: str = "09:00"
    end_time: str = "18:00"
    geo_center: Optional[Tuple[float, float]] = None
    geo_radius_km: float = 20.0
    indoor_preference: str = "neutral"  # "indoor" | "outdoor" | "neutral"

class POIResponse(BaseModel):
    id: int
    name: str
    lat: float
    lon: float
    category: str
    subcategory: str
    duration_hours: float
    cost: float
    opening_hours: str
    indoor_outdoor: str
    score: float

class EventResponse(BaseModel):
    poi: POIResponse
    start_time: str
    end_time: str
    travel_time_hours: float
    reason: str

class ItineraryResponse(BaseModel):
    itinerary: Dict[int, List[EventResponse]]
    total_cost: float
    total_days: int

# ---------- Initialize model once on startup ----------

EXCEL_FILE = "Cairo_Giza_POI_Database_v3.xlsx"

# Try to find the Excel file
if not os.path.exists(EXCEL_FILE):
    parent_file = os.path.join("..", EXCEL_FILE)
    if os.path.exists(parent_file):
        EXCEL_FILE = parent_file
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_file_abs = os.path.join(script_dir, "..", "Cairo_Giza_POI_Database_v3.xlsx")
        if os.path.exists(parent_file_abs):
            EXCEL_FILE = parent_file_abs

tourist_system = None

@app.on_event("startup")
async def startup_event():
    global tourist_system
    try:
        print(f"Initializing TouristRecommendationSystem with file: {EXCEL_FILE}")
        tourist_system = TouristRecommendationSystem(EXCEL_FILE)
        print("System initialized successfully!")
    except Exception as e:
        print(f"ERROR: Failed to initialize system: {e}")
        raise

# ---------- Endpoints ----------

@app.get("/")
def root():
    return {
        "message": "Nile Quest Recommendation API",
        "version": "1.0.0",
        "endpoints": {
            "POST /generate-itinerary": "Generate a personalized itinerary",
            "GET /health": "Health check"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "system_initialized": tourist_system is not None
    }

@app.post("/generate-itinerary", response_model=ItineraryResponse)
def generate_itinerary(req: UserProfileRequest):
    if tourist_system is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        # Create UserProfile from request
        user = UserProfile(
            interests=req.interests,
            budget_tier=req.budget_tier,
            duration_days=req.duration_days,
            pace=req.pace,
            start_time=req.start_time,
            end_time=req.end_time,
            geo_center=req.geo_center,
            geo_radius_km=req.geo_radius_km,
            indoor_preference=req.indoor_preference,
        )

        print(f"\nGenerating itinerary for user:")
        print(f"  Interests: {user.interests}")
        print(f"  Budget: {user.budget_tier} ({user.budget_daily} EGP/day)")
        print(f"  Duration: {user.duration_days} days")
        print(f"  Pace: {user.pace}")

        # Generate itinerary
        itinerary = tourist_system.generate_itinerary(user)

        # Convert to JSON-friendly structure and calculate total cost
        response_itinerary: Dict[int, List[EventResponse]] = {}
        total_cost = 0.0
        
        for day, events in itinerary.items():
            day_events: List[EventResponse] = []
            for event in events:
                poi = event["poi"]
                total_cost += poi.cost
                
                day_events.append(EventResponse(
                    poi=POIResponse(
                        id=poi.id,
                        name=poi.name,
                        lat=poi.lat,
                        lon=poi.lon,
                        category=poi.category,
                        subcategory=poi.subcategory,
                        duration_hours=poi.duration_hours,
                        cost=poi.cost,
                        opening_hours=poi.opening_hours,
                        indoor_outdoor=poi.indoor_outdoor,
                        score=poi.score,
                    ),
                    start_time=event["start_time"],
                    end_time=event["end_time"],
                    travel_time_hours=event["travel_time_hours"],
                    reason=event["reason"],
                ))
            response_itinerary[day] = day_events

        print(f"\nGenerated itinerary with {len(response_itinerary)} days, total cost: {total_cost} EGP")

        return ItineraryResponse(
            itinerary=response_itinerary,
            total_cost=total_cost,
            total_days=len(response_itinerary)
        )
        
    except Exception as e:
        print(f"ERROR generating itinerary: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate itinerary: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("Starting Nile Quest API Server...")
    print(f"Excel file path: {EXCEL_FILE}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
