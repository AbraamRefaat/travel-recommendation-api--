from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Tuple
import uvicorn
from tourist_recommendation_system import TouristRecommendationSystem, UserProfile

# Initialize FastAPI app
app = FastAPI(
    title="Tourist Recommendation API",
    description="API for generating personalized tourist itineraries",
    version="1.0.0"
)

# Add CORS middleware to allow Flutter app to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global system instance (initialized on startup)
recommendation_system: Optional[TouristRecommendationSystem] = None

# --- Pydantic Models for API ---

class UserProfileRequest(BaseModel):
    """Request model matching UserProfile structure"""
    interests: Dict[str, float] = Field(
        default_factory=dict,
        example={"History": 0.8, "Food": 0.5, "Nature": 0.6}
    )
    budget_daily: float = Field(default=1000.0, ge=0)
    budget_tier: str = Field(default="moderate", pattern="^(budget|moderate|luxury)$")
    budget_total: float = Field(default=5000.0, ge=0)
    duration_days: int = Field(default=1, ge=1, le=30)
    pace: str = Field(default="moderate", pattern="^(relaxed|moderate|packed)$")
    start_time: str = Field(default="09:00", pattern="^([01]?[0-9]|2[0-3]):[0-5][0-9]$")
    end_time: str = Field(default="17:00", pattern="^([01]?[0-9]|2[0-3]):[0-5][0-9]$")
    geo_center: Optional[Tuple[float, float]] = Field(
        default=None,
        example=[30.0444, 31.2357]
    )
    geo_radius_km: float = Field(default=20.0, ge=0)
    willingness_to_pay_entry: bool = Field(default=True)
    indoor_preference: str = Field(
        default="neutral",
        pattern="^(indoor|outdoor|neutral)$"
    )

class POIResponse(BaseModel):
    """Response model for a single POI"""
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
    description: str
    score: float

class ItineraryEventResponse(BaseModel):
    """Response model for a single event in the itinerary"""
    poi: POIResponse
    start_time: str
    end_time: str
    travel_time_hours: float
    reason: str

class ItineraryResponse(BaseModel):
    """Complete itinerary response"""
    success: bool
    itinerary: Dict[int, List[ItineraryEventResponse]]
    summary: Dict[str, any] = {}

# --- Startup Event ---

@app.on_event("startup")
async def startup_event():
    """Initialize the recommendation system on server startup"""
    global recommendation_system
    try:
        print("Initializing Tourist Recommendation System...")
        recommendation_system = TouristRecommendationSystem(
            "Cairo_Giza_POI_Database_v3.xlsx"
        )
        print("✓ System initialized successfully!")
    except Exception as e:
        print(f"✗ Failed to initialize system: {e}")
        raise

# --- Helper Functions ---

def poi_to_dict(poi) -> dict:
    """Convert POI object to dictionary"""
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

def event_to_dict(event: dict) -> dict:
    """Convert event dictionary (with POI object) to serializable format"""
    return {
        "poi": poi_to_dict(event["poi"]),
        "start_time": event["start_time"],
        "end_time": event["end_time"],
        "travel_time_hours": event["travel_time_hours"],
        "reason": event["reason"]
    }

# --- API Endpoints ---

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "message": "Tourist Recommendation API is running",
        "system_ready": recommendation_system is not None
    }

@app.post("/recommend", response_model=ItineraryResponse)
async def generate_recommendation(user_profile: UserProfileRequest):
    """
    Generate a personalized itinerary based on user preferences
    
    Args:
        user_profile: User preferences and constraints
        
    Returns:
        ItineraryResponse: Complete itinerary with POIs organized by day
    """
    if recommendation_system is None:
        raise HTTPException(
            status_code=503,
            detail="Recommendation system not initialized"
        )
    
    try:
        # Convert Pydantic model to UserProfile dataclass
        user = UserProfile(
            interests=user_profile.interests,
            budget_daily=user_profile.budget_daily,
            budget_tier=user_profile.budget_tier,
            budget_total=user_profile.budget_total,
            duration_days=user_profile.duration_days,
            pace=user_profile.pace,
            start_time=user_profile.start_time,
            end_time=user_profile.end_time,
            geo_center=user_profile.geo_center,
            geo_radius_km=user_profile.geo_radius_km,
            willingness_to_pay_entry=user_profile.willingness_to_pay_entry,
            indoor_preference=user_profile.indoor_preference
        )
        
        # Generate itinerary
        print(f"Generating itinerary for {user_profile.duration_days} days...")
        raw_itinerary = recommendation_system.generate_itinerary(user)
        
        # Convert to serializable format
        serializable_itinerary = {}
        total_cost = 0.0
        total_pois = 0
        
        for day, events in raw_itinerary.items():
            serializable_itinerary[day] = [event_to_dict(event) for event in events]
            
            # Calculate summary stats
            for event in events:
                total_cost += event["poi"].cost
                total_pois += 1
        
        # Build summary
        summary = {
            "total_days": user_profile.duration_days,
            "total_pois": total_pois,
            "total_cost_egp": total_cost,
            "daily_budget": user_profile.budget_daily,
            "budget_remaining": (user_profile.budget_daily * user_profile.duration_days) - total_cost
        }
        
        return {
            "success": True,
            "itinerary": serializable_itinerary,
            "summary": summary
        }
        
    except Exception as e:
        print(f"Error generating recommendation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate itinerary: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "system_initialized": recommendation_system is not None,
        "pois_loaded": len(recommendation_system.loader.pois) if recommendation_system else 0
    }

# --- Run Server ---

if __name__ == "__main__":
    print("Starting Tourist Recommendation API Server...")
    print("API Documentation will be available at: http://127.0.0.1:8000/docs")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",  # Listen on all interfaces
        port=8000,
        reload=True  # Enable auto-reload during development
    )
