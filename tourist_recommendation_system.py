
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import re
import math

# --- Data Structures ---

@dataclass
class UserProfile:
    """
    Stores user preferences and constraints.
    """
    interests: Dict[str, float] = field(default_factory=dict)  # {"History": 0.8, "Food": 0.5}
    budget_daily: float = 1000.0
    budget_tier: str = "moderate" # "budget", "moderate", "luxury"
    budget_total: float = 5000.0 # Optional global limit
    duration_days: int = 1
    pace: str = "moderate" # "relaxed", "moderate", "packed"
    start_time: str = "09:00"
    end_time: str = "17:00"
    geo_center: Optional[Tuple[float, float]] = None # (lat, lon)
    geo_radius_km: float = 20.0
    willingness_to_pay_entry: bool = True
    indoor_preference: str = "neutral" # "indoor", "outdoor", "neutral"

    def __post_init__(self):
        # Map budget_tier to budget_daily if not explicitly overridden by a custom value
        # (Assuming the default 1000.0 is a placeholder, or we just overwrite it based on tier)
        # To be safe, let's enforce tier Logic if it seems like a default.
        
        tier_map = {
            "budget": 1500.0,
            "moderate": 3500.0,
            "luxury": 10000.0
        }
        
        # If budget_daily is the default 1000.0, update it based on tier
        if self.budget_daily == 1000.0 and self.budget_tier.lower() in tier_map:
            self.budget_daily = tier_map[self.budget_tier.lower()]
            
        # Update total budget estimate
        if self.budget_total == 5000.0:
             self.budget_total = self.budget_daily * self.duration_days * 1.5

        # Normalize pace
        self.pace = self.pace.lower()

@dataclass
class POI:
    """
    Represents a single Point of Interest.
    """
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
    description: str = ""
    score: float = 0.0 # Dynamic score for the current user
    distance_from_last: float = 0.0 # Dynamic distance

# --- Components ---

class DataLoader:
    """
    Handles loading and cleaning of the Excel dataset.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None
        self.pois: List[POI] = []

    def load_data(self):
        print(f"Loading data from {self.file_path}...")
        try:
            self.df = pd.read_excel(self.file_path)
            # Standardize columns based on verified names
            # ['Name', 'Latitude / Longitude', 'Category', 'Sub-category', 'Estimated visit duration', 'Entry cost (EGP)', 'Opening hours', 'Indoor / outdoor']
            
            # Rename for easier access
            self.df.rename(columns={
                'Latitude / Longitude': 'Coordinates',
                'Estimated visit duration': 'Duration',
                'Entry cost (EGP)': 'Cost',
                'Opening hours': 'Hours',
                'Indoor / outdoor': 'Type'
            }, inplace=True)

            self._parse_pois()
            print(f"Successfully loaded {len(self.pois)} POIs.")
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def _parse_pois(self):
        for idx, row in self.df.iterrows():
            # Parse Coordinates
            lat, lon = self._parse_coordinates(row.get('Coordinates'))
            
            if lat is None or lon is None:
                continue # Skip invalid coordinates

            # Parse Duration (handle "2 hours", "1.5", etc.)
            duration = self._parse_duration(row.get('Duration'))

            # Parse Cost
            cost = pd.to_numeric(row.get('Cost'), errors='coerce') or 0.0
            
            # FORCE FOOD COST TO 0 (User Request: Variable cost, so assume 0 for planning)
            category = str(row.get('Category', 'General'))
            if category.lower() == 'food':
                cost = 0.0

            poi = POI(
                id=int(row.get('ID', idx)),
                name=str(row.get('Name', 'Unknown')),
                lat=lat,
                lon=lon,
                category=category,
                subcategory=str(row.get('Sub-category', '')),
                duration_hours=duration,
                cost=cost,
                opening_hours=str(row.get('Hours', '09:00 - 17:00')),
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
        # Default to 1.5 hours if unknown
        if pd.isna(duration_val):
            return 1.5
        try:
            # If it's a string like "2 hours", extract number
            val_str = str(duration_val).lower()
            match = re.search(r"(\d+(\.\d+)?)", val_str)
            if match:
                return float(match.group(1))
            return 1.5
        except:
            return 1.5

class CandidateGenerator:
    """
    Selects relevant POIs based on hard constraints and basic matching.
    """
    def __init__(self, all_pois: List[POI]):
        self.all_pois = all_pois

    def filter_candidates(self, user: UserProfile) -> List[POI]:
        candidates = []
        for poi in self.all_pois:
            # 1. Budget Hard Constraint (Individual POI check - optional, usually check daily limit later)
            # But let's check if a SINGLE ticket exceeds the ENTIRE daily budget (unlikely but possible)
            if poi.cost > user.budget_daily:
                continue
            
            if not user.willingness_to_pay_entry and poi.cost > 0:
                continue

            # 2. Indoor/Outdoor Constraint (if strict)
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

    def rank_pois(self, candidates: List[POI], user: UserProfile) -> List[POI]:
        for poi in candidates:
            score = 0.0
            
            # 1. Interest Match
            matched_interest = False
            for interest, weight in user.interests.items():
                if interest.lower() in poi.category.lower() or interest.lower() in poi.subcategory.lower():
                    score += weight * 10.0
                    matched_interest = True
                
                if interest.lower() in poi.description.lower():
                    score += weight * 2.0
            
            # Base Score
            if not matched_interest:
                 if "History" in poi.category or "Pharaonic" in poi.subcategory:
                     score += 2.0
                 else:
                     score += 1.0

            # 2. Cost Suitability (Smart Budget Logic)
            if user.budget_tier == "budget":
                if poi.cost < 100: score += 2.0
                elif poi.cost > 300: score -= 2.0
            elif user.budget_tier == "luxury":
                if poi.cost > 500: score += 2.0
                if poi.cost < 50: score -= 0.5
            else: # moderate
                 if 50 <= poi.cost <= 500: score += 1.0

            # 3. Duration Suitability based on Pace
            # "Relaxed": Penalize short/quick items to avoid rushing? Or penalize too many items?
            # Actually, for "Relaxed", we want FEWER items, so maybe we prefer longer, meaningful visits?
            # For "Packed", we want MANY items, so short duration is fine.
            
            if user.pace == "relaxed":
                # Prefer longer, deeper experiences
                if poi.duration_hours > 2.0: score += 1.0
                # Penalize quick stops slightly?
                if poi.duration_hours < 1.0: score -= 0.5
                
            elif user.pace == "packed":
                # Prefer shorter, quick hits
                if poi.duration_hours < 2.0: score += 1.0
                if poi.duration_hours > 4.0: score -= 1.0
            
            poi.score = score

        # Sort descending
        ranked = sorted(candidates, key=lambda x: x.score, reverse=True)
        return ranked

class ItineraryOptimizer:
    """
    Constructs a daily schedule ensuring time/budget constraints AND diversity.
    """
    def __init__(self):
        pass

    def _haversine(self, lat1, lon1, lat2, lon2):
        if lat1 is None or lat2 is None: return 0.0
        R = 6371
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) * math.sin(dlat / 2) + \
            math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * \
            math.sin(dlon / 2) * math.sin(dlon / 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def optimize_itinerary(self, ranked_candidates: List[POI], user: UserProfile) -> Dict[int, List[POI]]:
        """
        Smart Iterative Selection:
        - Pick best item.
        - Next item: Score = OriginalScore - DiversityPenalty - DistancePenalty
        """
        itinerary = {}
        
        # Track global usage
        total_cost_spent = 0.0
        used_poi_ids = set()
        
        # Start times
        try:
           h, m = map(int, user.start_time.split(':'))
           start_h_float = h + m/60.0
           endo_h, endo_m = map(int, user.end_time.split(':'))
           end_h_float = endo_h + endo_m/60.0
        except:
           start_h_float = 9.0
           end_h_float = 17.0

        for day in range(1, user.duration_days + 1):
            day_pois = []
            daily_cost_spent = 0.0
            current_time = start_h_float
            last_location = user.geo_center # Start from hotel/center

            # Track categories used TODAY for diversity
            day_category_counts = {}

            # Determine max items based on pace
            if user.pace == "relaxed":
                max_items = 2
            elif user.pace == "moderate":
                max_items = 4 # "Balanced like 3" - giving a bit of flex, or set to 3. Let's say 4 to allow small items.
                              # User said "Moderate to be balanced like 3". Let's stick to 3-4.
                max_items = 3 
            else: # packed
                max_items = 99

            # While we have time in the day AND space in the schedule
            while current_time < end_h_float and len(day_pois) < max_items:
                best_candidate = None
                best_effective_score = -float('inf')

                # Re-evaluate all valid candidates for this specific slot
                for poi in ranked_candidates:
                    if poi.id in used_poi_ids:
                        continue
                    
                    # Hard Constraints
                    if (daily_cost_spent + poi.cost) > user.budget_daily: continue
                    if (total_cost_spent + poi.cost) > user.budget_total: continue
                    
                    # Time Constraint (POIDuration + TravelBuffer)
                    # Estimate travel from LAST SELECTED location
                    dist_km = 0
                    if last_location:
                        dist_km = self._haversine(last_location[0], last_location[1], poi.lat, poi.lon)
                    
                    travel_time_h = max(0.25, dist_km / 20.0) # 20km/h avg
                    if (current_time + poi.duration_hours + travel_time_h) > end_h_float:
                        continue
                    
                    # --- SCORING ---
                    # 1. Base Score
                    effective_score = poi.score
                    
                    # 2. Diversity Penalty (Category Mix)
                    # If we already have a "Museum", penalize next matching category heavily
                    # Check broad category and subcategory
                    penalty = 0.0
                    for cat_str in [poi.category, poi.subcategory]:
                        # Simple keyword check against already used
                        for used_cat in day_category_counts:
                            if used_cat in cat_str or cat_str in used_cat:
                                penalty += 5.0 # Big penalty for repetition
                    
                    effective_score -= penalty

                    # 3. Distance Penalty (Smart Routing)
                    # Penalize far items to encourage clustering
                    # e.g. -0.5 score per km
                    effective_score -= (dist_km * 0.5)

                    if effective_score > best_effective_score:
                        best_effective_score = effective_score
                        best_candidate = poi

                # If no candidate found, break for the day
                if not best_candidate:
                    break
                
                # Add best candidate
                day_pois.append(best_candidate)
                used_poi_ids.add(best_candidate.id)
                
                # Update State
                daily_cost_spent += best_candidate.cost
                total_cost_spent += best_candidate.cost
                
                # Update Time
                dist_km = 0
                if last_location:
                    dist_km = self._haversine(last_location[0], last_location[1], best_candidate.lat, best_candidate.lon)
                travel_h = max(0.25, dist_km / 20.0)
                current_time += (best_candidate.duration_hours + travel_h)
                
                # Update Location
                last_location = (best_candidate.lat, best_candidate.lon)
                
                # Update Diversity Counts
                cat_key = f"{best_candidate.category}|{best_candidate.subcategory}"
                day_category_counts[cat_key] = day_category_counts.get(cat_key, 0) + 1
            
            itinerary[day] = day_pois

        return itinerary

class Scheduler:
    """
    Refines the daily itinerary by ordering POIs geographically (TSP-Greedy).
    """
    def __init__(self):
        pass

    def _haversine(self, lat1, lon1, lat2, lon2):
        if lat1 is None or lat2 is None: return 0.0
        R = 6371  # Earth radius in km
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) * math.sin(dlat / 2) + \
            math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * \
            math.sin(dlon / 2) * math.sin(dlon / 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def route_day(self, pois: List[POI], start_coords: Tuple[float, float] = None) -> List[POI]:
        """
        Orders the POIs to minimize travel distance.
        """
        if not pois:
            return []

        # Simple Greedy TSP
        ordered_pois = []
        # If no start location provided and we just picked first POI, use it as start
        remaining = pois[:]
        
        if start_coords:
             current_location = start_coords
        else:
             # Pick the highest ranked (first in list) as the anchor if no start
             first = remaining.pop(0)
             ordered_pois.append(first)
             current_location = (first.lat, first.lon)
        
        while remaining:
            # Find nearest to current_location
            nearest_poi = min(remaining, key=lambda p: self._haversine(current_location[0], current_location[1], p.lat, p.lon))
            ordered_pois.append(nearest_poi)
            current_location = (nearest_poi.lat, nearest_poi.lon)
            remaining.remove(nearest_poi)
            
        return ordered_pois

    def generate_timed_itinerary(self, optimized_plan: Dict[int, List[POI]], user: UserProfile) -> Dict[int, List[dict]]:
        final_itinerary = {}
        
        # Parse start time: "09:00" -> 9.0
        try:
            h, m = map(int, user.start_time.split(':'))
            start_h_float = h + m/60.0
        except:
            start_h_float = 9.0
        
        for day, pois in optimized_plan.items():
            if not pois:
                final_itinerary[day] = []
                continue

            # 1. Route
            routed_pois = self.route_day(pois, user.geo_center)
            
            # 2. Assign Times
            day_schedule = []
            current_time = start_h_float
            last_coords = user.geo_center
            
            for i, poi in enumerate(routed_pois):
                # Calc travel from last
                travel_time = 0.0
                if last_coords:
                     dist = self._haversine(last_coords[0], last_coords[1], poi.lat, poi.lon)
                     # Assume 20km/h avg speed in city traffic
                     hours = dist / 20.0
                     travel_time = max(0.25, min(hours, 1.5)) # clamp 15m to 1.5h
                elif i > 0:
                     # Fallback if no user center
                     pass
                
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
    def __init__(self, excel_file: str):
        self.loader = DataLoader(excel_file)
        self.loader.load_data()
        
        # Initialize AI Generator
        try:
            self.ai_gen = AICandidateGenerator(excel_file)
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
                ai_results = self.ai_gen.generate_candidates_for_user(user, top_k=100)
                
                # 2. Map back to POI objects using ID (Index)
                # Create a quick lookup map
                poi_map = {p.id: p for p in self.loader.pois}
                
                for idx, row in ai_results.iterrows():
                    if idx in poi_map:
                        poi = poi_map[idx]
                        # Inject AI Score
                        # Scale 0-1 to useful score, e.g. 0-10 base
                        poi.score = row.get('Semantic_Score', 0) * 100.0 
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
    sys = TouristRecommendationSystem("Cairo_Giza_POI_Database_v3.xlsx")
    
    # Define a test user: 3 days, likes History & Nature
    user = UserProfile(
        interests={"History": 0.9, "Nature": 0.6, "Nile": 0.4},
        budget_daily=2000.0,
        budget_total=6000.0,
        duration_days=3,
        pace="moderate",
        start_time="09:00",
        end_time = "19:00",
        geo_center=(30.0444, 31.2357) # Downtown Cairo approx
    )
    
    print("\nGenerating Itinerary for User Preferences:")
    print(f"Interests: {user.interests}")
    print(f"Days: {user.duration_days}, Budget/Day: {user.budget_daily}")
    
    itinerary = sys.generate_itinerary(user)
    
    print("\n" + "="*50)
    print("       FINAL SUGGESTED ITINERARY       ")
    print("="*50)
    
    total_cost = 0
    for day, events in itinerary.items():
        print(f"\n[ DAY {day} ]")
        day_cost = 0
        for event in events:
            p = event['poi']
            print(f"  {event['start_time']} - {event['end_time']} : {p.name}")
            print(f"     -> {p.category} | {p.subcategory}")
            print(f"     -> Cost: {p.cost} EGP")
            print(f"     -> Reason: {event['reason']}")
            day_cost += p.cost
        
        print(f"  --> Daily Total: {day_cost} EGP")
        total_cost += day_cost
        
    print("="*50)
    print(f"Trip Total Cost: {total_cost} EGP")

