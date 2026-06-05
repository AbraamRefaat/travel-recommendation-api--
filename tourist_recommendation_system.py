
import pandas as pd
import numpy as np
import os
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import re
import math

try:
    from .engine_config import CONFIG
except ImportError:
    from engine_config import CONFIG


# --- Shared geo / travel helpers (single source of truth) ---

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in km between two lat/lon points."""
    if None in (lat1, lon1, lat2, lon2):
        return 0.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return EARTH_RADIUS_KM * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def travel_hours(dist_km: float) -> float:
    """Convert a distance to a travel-time estimate using the configured speed/clamps."""
    raw = dist_km / CONFIG.travel_speed_kmh if CONFIG.travel_speed_kmh > 0 else 0.0
    return max(CONFIG.min_travel_hours, min(raw, CONFIG.max_travel_hours))


# --- Data Structures ---

def _tier_budget_map() -> Dict[str, float]:
    """Read daily budget limits per tier from env vars, with sensible fallbacks."""
    return {
        "budget":   float(os.environ.get("BUDGET_DAILY_BUDGET",   1500)),
        "moderate": float(os.environ.get("BUDGET_DAILY_MODERATE", 3500)),
        "luxury":   float(os.environ.get("BUDGET_DAILY_LUXURY",  10000)),
    }

def _default_geo_radius() -> float:
    return float(os.environ.get("DEFAULT_GEO_RADIUS_KM", 20.0))

@dataclass
class UserProfile:
    """
    Stores user preferences and constraints.
    """
    interests: Dict[str, float] = field(default_factory=dict)
    budget_tier: str = "moderate"           # "budget", "moderate", "luxury"
    budget_daily: Optional[float] = None    # derived from tier if not provided
    duration_days: int = 1
    pace: str = "moderate"                  # "relaxed", "moderate", "packed"
    geo_center: Optional[Tuple[float, float]] = None
    geo_radius_km: float = field(default_factory=_default_geo_radius)
    willingness_to_pay_entry: bool = True
    indoor_preference: str = "neutral"      # "indoor", "outdoor", "neutral"

    def __post_init__(self):
        self.budget_tier = self.budget_tier.lower()
        self.pace = self.pace.lower()
        if self.budget_daily is None:
            tier_map = _tier_budget_map()
            self.budget_daily = tier_map.get(self.budget_tier, tier_map["moderate"])

@dataclass
class POI:
    """
    Represents a single Point of Interest.
    """
    id: Union[int, str]   # can be int or UUID string depending on ingestion method
    name: str
    lat: float
    lon: float
    category: str
    subcategory: str
    duration_hours: float
    price_range: str     # raw value: "$", "$$", or "$$$"
    opening_hours: str
    indoor_outdoor: str
    description: str = ""
    score: float = 0.0
    distance_from_last: float = 0.0

    @staticmethod
    def from_payload(point_id, payload):
        """Construct a POI object dynamically from a Qdrant payload."""
        lat_lon = payload.get('Latitude / Longitude', '0,0')
        lat, lon = 0.0, 0.0
        try:
            parts = str(lat_lon).split(',')
            lat, lon = float(parts[0].strip()), float(parts[1].strip())
        except: pass

        duration = CONFIG.default_visit_hours
        try:
            d_str = str(payload.get('Estimated visit duration', ''))
            match = re.search(r"(\d+(\.\d+)?)", d_str)
            if match: duration = float(match.group(1))
        except: pass

        price_range = ""
        try:
            price_range = str(payload.get('Price range', '')).strip()
        except: pass

        open_t  = str(payload.get('Opening time', ''))
        close_t = str(payload.get('Closing time', ''))
        opening = f"{open_t} - {close_t}" if (open_t or close_t) else ''

        # Parse ID — support both integer IDs and UUID strings
        try:
            parsed_id = int(point_id)
        except (ValueError, TypeError):
            parsed_id = str(point_id)  # keep as UUID string

        return POI(
            id=parsed_id,
            name=str(payload.get('Name', 'Unknown')),
            lat=lat,
            lon=lon,
            category=str(payload.get('Category', 'Unknown')),
            subcategory=str(payload.get('Sub-category', '')),
            duration_hours=duration,
            price_range=price_range,
            opening_hours=opening,
            indoor_outdoor=str(payload.get('Indoor / outdoor', 'Both')),
            description=f"{payload.get('Category')} - {payload.get('Sub-category')}"
        )

# --- Components ---

class DataLoader:
    """
    Handles loading and cleaning of the Excel dataset.
    """
    def __init__(self, collection_name: str = "pois"):
        self.collection_name = collection_name
        self.df = None
        self.pois: List[POI] = []
        from qdrant_client import QdrantClient
        host = os.environ.get("QDRANT_HOST", "localhost")
        port = int(os.environ.get("QDRANT_PORT", 6333))
        api_key = os.environ.get("QDRANT_API_KEY")
        
        try:
            if api_key:
                print(f"📡 [DataLoader] Connecting to Qdrant at {host}:{port} with API key...")
                self.client = QdrantClient(
                    host=host, 
                    port=port, 
                    api_key=api_key,
                    https=(port == 443 or port == 6333)
                )
            else:
                print(f"📡 [DataLoader] Connecting to Qdrant at {host}:{port}...")
                self.client = QdrantClient(host=host, port=port, prefer_grpc=False)
        except Exception as e:
            print(f"❌ [DataLoader] Failed to connect to Qdrant: {e}")
            raise

    def load_data(self):
        """Loading data is now dynamic from Qdrant. No Excel needed."""
        print(f"📡 [DataLoader] Refreshing data from Qdrant collection '{self.collection_name}'...")
        try:
            self.pois = []
            # In a truly stateless app, we could skip pre-loading, but keeping it 
            # for the current ranker/optimizer logic which expects a base list.
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=CONFIG.qdrant_scroll_limit,
                with_payload=True,
                with_vectors=False
            )
            
            for p in points:
                self.pois.append(POI.from_payload(p.id, p.payload))
            
            print(f"✅ [DataLoader] Loaded {len(self.pois)} POIs from Qdrant.")
        except Exception as e:
            print(f"❌ [DataLoader] Error loading from Qdrant: {e}.")

    def _parse_pois(self):
        for idx, row in self.df.iterrows():
            # Parse Coordinates
            lat, lon = self._parse_coordinates(row.get('Coordinates'))
            
            if lat is None or lon is None:
                continue # Skip invalid coordinates

            # Parse Duration (handle "2 hours", "1.5", etc.)
            duration = self._parse_duration(row.get('Duration'))

            category = str(row.get('Category', 'General'))
            price_range = str(row.get('Price range', '')).strip()

            poi = POI(
                id=int(row.get('ID', idx)),
                name=str(row.get('Name', 'Unknown')),
                lat=lat,
                lon=lon,
                category=category,
                subcategory=str(row.get('Sub-category', '')),
                duration_hours=duration,
                price_range=price_range,
                opening_hours=str(row.get('Hours', '')),
                indoor_outdoor=str(row.get('Type', 'Both')),
                description=f"{row.get('Category')} - {row.get('Sub-category')}"
            )
            self.pois.append(poi)

    def _parse_coordinates(self, coord_str):
        try:
            if isinstance(coord_str, str):
                parts = coord_str.split(',')
                if len(parts) >= 2:
                    return float(parts[0].strip()), float(parts[1].strip())
            return None, None
        except:
            return None, None

    def _parse_duration(self, duration_val):
        if pd.isna(duration_val):
            return CONFIG.default_visit_hours
        try:
            val_str = str(duration_val).lower()
            match = re.search(r"(\d+(\.\d+)?)", val_str)
            if match:
                return float(match.group(1))
            return CONFIG.default_visit_hours
        except:
            return CONFIG.default_visit_hours

class CandidateGenerator:
    """
    Selects relevant POIs based on hard constraints and basic matching.
    """
    def __init__(self, all_pois: List[POI]):
        self.all_pois = all_pois

    def filter_candidates(self, user: UserProfile) -> List[POI]:
        candidates = []
        for poi in self.all_pois:
            # 1. Indoor/Outdoor Constraint (if strict)
            if user.indoor_preference != "neutral":
                # If user wants Indoor, skip purely Outdoor places
                # Data might say "Indoor", "Outdoor", "Both"
                poi_type = poi.indoor_outdoor.lower()
                if user.indoor_preference == "indoor" and "outdoor" in poi_type and "indoor" not in poi_type:
                    continue
                if user.indoor_preference == "outdoor" and "indoor" in poi_type and "outdoor" not in poi_type:
                    continue

            # 3. Geo Filter (Example: Simple radius check if center provided)
            # Omitted for now to keep it broad, usually filtering happen in ranking or routing
            
            # 4. Interest Matching (Semantic/Keyword) 
            # We will calculate a base score here or just include everything that matches AT LEAST one category
            # For now, we are permissive: include almost everything, let Ranking sort them.
            # But we can remove things that explicitly don't match any interest if the user provided specific ones.
            
            candidates.append(poi)
        

        print(f"Candidate Generation: Reduced {len(self.all_pois)} to {len(candidates)} candidates.")
        return candidates


class POIRanker:
    """
    Ranks candidates based on user interests and other factors.
    """
    def __init__(self):
        pass

    PRICE_ORDER = {"$": 1, "$$": 2, "$$$": 3}

    def rank_pois(self, candidates: List[POI], user: UserProfile) -> List[POI]:
        if not candidates:
            return []

        # --- Preserve the upstream AI/semantic relevance instead of discarding it. ---
        # The AI generator stores its relevance in poi.score; normalise it to 0..1 so it
        # can be blended on a comparable scale with the heuristic signals below.
        semantic_raw = [p.score for p in candidates]
        lo, hi = min(semantic_raw), max(semantic_raw)
        span = (hi - lo) or 1.0

        # Median visit duration drives the data-relative pace preference (no fixed thresholds).
        durations = sorted(p.duration_hours for p in candidates)
        median_duration = durations[len(durations) // 2]

        for poi in candidates:
            semantic_norm = (poi.score - lo) / span  # 0..1
            score = CONFIG.w_semantic * semantic_norm

            # 1. Explicit interest match (category/subcategory/description keyword overlap)
            for interest, weight in user.interests.items():
                il = interest.lower()
                if il in poi.category.lower() or il in poi.subcategory.lower():
                    score += weight * CONFIG.w_interest_category
                if il in poi.description.lower():
                    score += weight * CONFIG.w_interest_description

            # Neutral base so an item with no explicit keyword hit still survives on
            # its semantic merit (no city- or theme-specific bias hardcoded here).
            score += CONFIG.base_score_no_match

            # 2. Price-tier suitability relative to the user's budget tier
            pr = self.PRICE_ORDER.get(poi.price_range, 0)
            if pr:
                if user.budget_tier == "budget":
                    if pr == 1:   score += CONFIG.price_match_bonus
                    elif pr == 2: score += CONFIG.price_tolerate_bonus
                    else:         score -= CONFIG.price_mismatch_penalty
                elif user.budget_tier == "luxury":
                    if pr == 3:   score += CONFIG.price_match_bonus
                    elif pr == 2: score += CONFIG.price_tolerate_bonus
                else:  # moderate
                    if pr <= 2:   score += CONFIG.price_tolerate_bonus

            # 3. Pace suitability — measured against the dataset's own median duration,
            #    so "long" / "short" adapt to whatever data is loaded.
            if user.pace == "relaxed" and poi.duration_hours >= median_duration:
                score += CONFIG.price_tolerate_bonus
            elif user.pace == "packed" and poi.duration_hours <= median_duration:
                score += CONFIG.price_tolerate_bonus

            poi.score = score

        return sorted(candidates, key=lambda x: x.score, reverse=True)

class ItineraryOptimizer:
    """
    Constructs a daily schedule ensuring time constraints AND diversity.
    """
    def __init__(self):
        pass

    def _get_day_window(self, pois: List[POI]):
        """Derive day start/end from POIs' actual opening hours."""
        opens, closes = [], []
        for poi in pois:
            try:
                parts = poi.opening_hours.split(' - ')
                oh, om = map(int, parts[0].strip().split(':'))
                opens.append(oh + om / 60.0)
                ch, cm = map(int, parts[1].strip().split(':'))
                closes.append(ch + cm / 60.0)
            except Exception:
                pass
        start = min(opens) if opens else CONFIG.default_day_start
        end = max(closes) if closes else CONFIG.default_day_end
        return start, end

    def _max_items_for_day(self, pace: str) -> int:
        """Fixed number of stops per day, driven by pace (overridable via env)."""
        return CONFIG.max_items.get(pace, CONFIG.max_items["moderate"])

    def optimize_itinerary(self, ranked_candidates: List[POI], user: UserProfile) -> Dict[int, List[POI]]:
        """
        Smart Iterative Selection:
        - Pick best item.
        - Next item: Score = OriginalScore - DiversityPenalty - DistancePenalty
        """
        itinerary = {}
        used_poi_ids = set()

        start_h_float, end_h_float = self._get_day_window(ranked_candidates)
        max_items = self._max_items_for_day(user.pace)

        for day in range(1, user.duration_days + 1):
            day_pois = []
            current_time = start_h_float
            last_location = user.geo_center  # Start from hotel/center

            # Track categories used TODAY for diversity
            day_category_counts = {}

            # While we have time in the day AND space in the schedule
            while current_time < end_h_float and len(day_pois) < max_items:
                best_candidate = None
                best_dist = 0.0
                best_effective_score = -float('inf')

                # Re-evaluate all valid candidates for this specific slot
                for poi in ranked_candidates:
                    if poi.id in used_poi_ids:
                        continue

                    # Time Constraint (visit duration + travel buffer from last stop)
                    dist_km = 0.0
                    if last_location:
                        dist_km = haversine_km(last_location[0], last_location[1], poi.lat, poi.lon)

                    if (current_time + poi.duration_hours + travel_hours(dist_km)) > end_h_float:
                        continue

                    # --- SCORING ---
                    effective_score = poi.score

                    # Diversity penalty: escalates the more we repeat a category today.
                    cat_key = f"{poi.category}|{poi.subcategory}"
                    repeats = day_category_counts.get(cat_key, 0)
                    effective_score -= repeats * CONFIG.diversity_penalty

                    # Distance penalty encourages geographic clustering.
                    effective_score -= dist_km * CONFIG.distance_penalty_per_km

                    if effective_score > best_effective_score:
                        best_effective_score = effective_score
                        best_candidate = poi
                        best_dist = dist_km

                if not best_candidate:
                    break

                day_pois.append(best_candidate)
                used_poi_ids.add(best_candidate.id)

                current_time += best_candidate.duration_hours + travel_hours(best_dist)
                last_location = (best_candidate.lat, best_candidate.lon)

                cat_key = f"{best_candidate.category}|{best_candidate.subcategory}"
                day_category_counts[cat_key] = day_category_counts.get(cat_key, 0) + 1

            itinerary[day] = day_pois

        return itinerary

class Scheduler:
    """
    Refines the daily itinerary by ordering POIs geographically
    (greedy nearest-neighbour seed + 2-opt improvement).
    """
    def __init__(self):
        pass

    def _route_length(self, coords: List[Tuple[float, float]], start: Tuple[float, float] = None) -> float:
        total = 0.0
        prev = start
        for c in coords:
            if prev is not None:
                total += haversine_km(prev[0], prev[1], c[0], c[1])
            prev = c
        return total

    def _two_opt(self, pois: List[POI], start: Tuple[float, float] = None) -> List[POI]:
        """Improve a route by repeatedly reversing segments that shorten total travel."""
        if len(pois) < 4:
            return pois
        route = pois[:]
        coords = lambda lst: [(p.lat, p.lon) for p in lst]
        improved = True
        while improved:
            improved = False
            best = self._route_length(coords(route), start)
            for i in range(len(route) - 1):
                for k in range(i + 1, len(route)):
                    candidate = route[:i] + route[i:k + 1][::-1] + route[k + 1:]
                    length = self._route_length(coords(candidate), start)
                    if length + 1e-9 < best:
                        route, best, improved = candidate, length, True
        return route

    def route_day(self, pois: List[POI], start_coords: Tuple[float, float] = None) -> List[POI]:
        """Order POIs to minimize travel distance (greedy seed, then 2-opt)."""
        if not pois:
            return []

        # 1. Greedy nearest-neighbour seed
        ordered_pois = []
        remaining = pois[:]
        if start_coords:
            current_location = start_coords
        else:
            first = remaining.pop(0)
            ordered_pois.append(first)
            current_location = (first.lat, first.lon)

        while remaining:
            nearest_poi = min(
                remaining,
                key=lambda p: haversine_km(current_location[0], current_location[1], p.lat, p.lon),
            )
            ordered_pois.append(nearest_poi)
            current_location = (nearest_poi.lat, nearest_poi.lon)
            remaining.remove(nearest_poi)

        # 2. 2-opt refinement
        return self._two_opt(ordered_pois, start_coords)

    def _parse_open_time(self, opening_hours: str) -> float:
        """Parse opening time from 'HH:MM - HH:MM' string, return float hours."""
        try:
            h, m = map(int, opening_hours.split(' - ')[0].strip().split(':'))
            return h + m / 60.0
        except Exception:
            return CONFIG.default_day_start

    def generate_timed_itinerary(self, optimized_plan: Dict[int, List[POI]], user: UserProfile) -> Dict[int, List[dict]]:
        final_itinerary = {}

        for day, pois in optimized_plan.items():
            if not pois:
                final_itinerary[day] = []
                continue

            # 1. Route
            routed_pois = self.route_day(pois, user.geo_center)

            # 2. Assign Times — start from the earliest opening time among this day's POIs
            day_schedule = []
            if routed_pois:
                start_h_float = min(self._parse_open_time(p.opening_hours) for p in routed_pois)
            else:
                start_h_float = CONFIG.default_day_start
            current_time = start_h_float
            last_coords = user.geo_center

            for i, poi in enumerate(routed_pois):
                # Calc travel from last
                travel_time = 0.0
                if last_coords:
                     dist = haversine_km(last_coords[0], last_coords[1], poi.lat, poi.lon)
                     travel_time = travel_hours(dist)

                # Arrival
                arrival_time = current_time + travel_time
                end_time = arrival_time + poi.duration_hours
                
                # Format times
                def fmt(t):
                    h = int(t)
                    m = int((t - h) * 60)
                    if m >= 60: 
                        h += 1
                        m -= 60
                    return f"{h:02}:{m:02}"

                matched_int = [k for k in user.interests if k.lower() in poi.category.lower() or k.lower() in poi.subcategory.lower()]
                reason_str = f"Matches {', '.join(matched_int)}" if matched_int else "Popular attraction"

                day_schedule.append({
                    "poi": poi,
                    "start_time": fmt(arrival_time),
                    "end_time": fmt(end_time),
                    "travel_time_hours": travel_time,
                    "reason": f"{reason_str} (Score: {poi.score:.1f})"
                })
                
                current_time = end_time
                last_coords = (poi.lat, poi.lon)
            
            final_itinerary[day] = day_schedule
            
        return final_itinerary

try:
    from .ai_candidate_generator import AICandidateGenerator
except ImportError:
    from ai_candidate_generator import AICandidateGenerator

class TouristRecommendationSystem:
    def __init__(self, collection_name: str = "pois", model=None):
        self.loader = DataLoader(collection_name)
        self.loader.load_data()
        
        # Initialize AI Generator
        try:
            self.ai_gen = AICandidateGenerator(collection_name, shared_model=model)
            self.use_ai = True
        except Exception as e:
            print(f"Warning: Could not initialize AI model ({e}). Falling back to basic filter.")
            self.use_ai = False
            
        self.candidate_gen = CandidateGenerator(self.loader.pois) # Keep as fallback
        self.ranker = POIRanker()
        self.optimizer = ItineraryOptimizer()
        self.scheduler = Scheduler()

    def generate_itinerary(self, user: UserProfile):
        candidates = []
        
        if self.use_ai:
            print("1. AI Filtering Candidates (Semantic Search)...")
            try:
                # 1. Get Top Candidates from AI
                # This now returns Qdrant hits with payloads included
                ai_results_df = self.ai_gen.generate_candidates_for_user(user, top_k=100)
                
                for _, row in ai_results_df.iterrows():
                    # Construct POI directly from the Qdrant payload results
                    poi = POI.from_payload(row['id'], row)
                    # Inject AI Score
                    raw_semantic_score = row.get('Semantic_Score', 0)
                    poi.score = float(raw_semantic_score) * 100.0 if pd.notna(raw_semantic_score) else 0.0
                    candidates.append(poi)
                        
                print(f"AI returned {len(candidates)} valid candidates.")
                
            except Exception as e:
                print(f"AI Generation failed: {e}. Using basic filter.")
                candidates = self.candidate_gen.filter_candidates(user)
        else:
            print("1. Basic Filtering Candidates...")
            candidates = self.candidate_gen.filter_candidates(user)
        
        # Fallback if AI found nothing (e.g. constraints too strict)
        if not candidates:
             print("AI found no matches, retrying with basic loose filter...")
             candidates = self.candidate_gen.filter_candidates(user)

        print("2. Ranking...")
        # We can still run the ranker to apply specific heuristics (like 'History' keyword bonus)
        # or just rely on the AI score. Let's start with a fresh rank to be safe, 
        # but maybe we should preserve the AI score as a base?
        # The Ranker currently overwrites poi.score. 
        # Let's modify the Ranker usage or trust the Ranker to do a good job on the Reduced set.
        # Actually, let's just let the Ranker refine the AI's selection.
        ranked = self.ranker.rank_pois(candidates, user)
        
        print("3. Optimization...")
        optimized_days = self.optimizer.optimize_itinerary(ranked, user)
        
        print("4. Scheduling...")
        final_schedule = self.scheduler.generate_timed_itinerary(optimized_days, user)
        
        return final_schedule

if __name__ == "__main__":
    # Test Run
    print("Initializing System...")
    sys = TouristRecommendationSystem("pois")
    
    # Define a test user: 3 days, likes History & Nature
    user = UserProfile(
        interests={"History": 0.9, "Nature": 0.6, "Nile": 0.4},
        budget_daily=2000.0,
        duration_days=3,
        pace="moderate",
        geo_center=(30.0444, 31.2357) # Downtown Cairo approx
    )
    
    print("\nGenerating Itinerary for User Preferences:")
    print(f"Interests: {user.interests}")
    print(f"Days: {user.duration_days}, Budget/Day: {user.budget_daily}")
    
    itinerary = sys.generate_itinerary(user)
    
    print("\n" + "="*50)
    print("       FINAL SUGGESTED ITINERARY       ")
    print("="*50)
    
    for day, events in itinerary.items():
        print(f"\n[ DAY {day} ]")
        for event in events:
            p = event['poi']
            print(f"  {event['start_time']} - {event['end_time']} : {p.name}")
            print(f"     -> {p.category} | {p.subcategory}")
            print(f"     -> Price range: {p.price_range}")
            print(f"     -> Opening: {p.opening_hours}")
            print(f"     -> Reason: {event['reason']}")
    print("="*50)

