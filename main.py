
import os
import sys

# Add parent directory to path if running directly to find siblings if needed, 
# but here we want to run as a script in the folder.
# We need to handle the import of the sibling module.

try:
    from tourist_recommendation_system import TouristRecommendationSystem, UserProfile
except ImportError:
    # If running from parent directory as a module e.g. python -m recommendation_model.main
    from .tourist_recommendation_system import TouristRecommendationSystem, UserProfile

def main():
    print("=== AI Tourist Recommendation System ===")
    
    # 1. Initialize System
    excel_file = "Cairo_Giza_POI_Database_v3.xlsx"
    # Check if file exists in current dir, else check parent
    if not os.path.exists(excel_file):
        parent_file = os.path.join("..", excel_file)
        if os.path.exists(parent_file):
            excel_file = parent_file
        else:
             # Try absolute path based on script location
             script_dir = os.path.dirname(os.path.abspath(__file__))
             parent_file_abs = os.path.join(script_dir, "..", "Cairo_Giza_POI_Database_v3.xlsx")
             if os.path.exists(parent_file_abs):
                 excel_file = parent_file_abs

    try:
        sys_instance = TouristRecommendationSystem(excel_file)
    except Exception as e:
        print(f"Failed to initialize: {e}")
        return


    # 2. Define User Profile (Interactive Input)
    print("\n--- Enter Your Preferences ---")
    
    # Duration
    while True:
        try:
            days = int(input("How many days is your trip? (e.g., 3): ").strip() or "1")
            if days > 0: break
            print("Please enter a positive number.")
        except ValueError:
            print("Invalid number.")

    # Budget
    print("\nSelect your Budget Tier:")
    budget_options = ["Budget", "Moderate", "Luxury"]
    for i, opt in enumerate(budget_options, 1):
        print(f"{i}. {opt}")
        
    budget_tier = "moderate"
    while True:
        try:
            choice = int(input("Select option (1-3): ").strip())
            if 1 <= choice <= 3:
                budget_tier = budget_options[choice-1].lower()
                break
            print("Invalid choice.")
        except ValueError:
            print("Please enter a number.")

    # Interests
    print("\nSelect your top interests BY PRIORITY (1st is most important).")
    available_interests = ["History", "Food", "Nature", "Shopping", "Entertainment", "Religious"]
    
    print("Available options:")
    for i, opt in enumerate(available_interests, 1):
        print(f" {i}. {opt}")
        
    user_interests = {}
    
    # Dynamic Priority Input Loop
    print("\nSelect your top interests BY PRIORITY.")
    print("1st choice is most important, 2nd is next, etc.")
    print("Press Enter without typing to finish.")
    
    current_priority = 1
    max_weight = 1.0
    
    while True:
        prompt = f"\n{current_priority}. What is your next interest? "
        if current_priority == 1:
            prompt = "\n1. What is your MAIN interest? "
            
        choice = input(prompt + "(Enter name): ").strip().title()
        
        if not choice:
            if current_priority == 1:
                 print("   Please enter at least one interest.")
                 continue
            else:
                 break # User finished
                 
        if choice in available_interests:
            if choice not in user_interests:
                # Assign weight: 1.0, 0.9, 0.8 ... min 0.1
                weight = max(0.1, max_weight - (current_priority - 1) * 0.1)
                user_interests[choice] = weight
                current_priority += 1
            else:
                print(f"   '{choice}' is already selected.")
        else:
            print(f"   Please choose from the list above: {', '.join(available_interests)}")

    # Custom interest (Added as the next priority)
    custom = input("\nAny specific topic not listed? (e.g. 'Pharaonic'): ").strip()
    if custom:
        weight = max(0.1, max_weight - (current_priority - 1) * 0.1)
        user_interests[custom] = weight # Add to the tail of priorities

    if not user_interests:
        print("No specific interests provided. We will recommend popular attractions.")
    
    # Pace
    print("\nSelect your Travel Pace:")
    pace_options = ["Relaxed", "Moderate", "Packed"]
    for i, opt in enumerate(pace_options, 1):
        print(f"{i}. {opt}")
        
    pace = "moderate"
    while True:
        try:
            choice = int(input("Select option (1-3): ").strip())
            if 1 <= choice <= 3:
                pace = pace_options[choice-1].lower()
                break
            print("Invalid choice.")
        except ValueError:
             print("Please enter a number.")

    
    user = UserProfile(
        interests=user_interests,
        budget_tier=budget_tier,
        # budget_daily and budget_total will be estimated in __post_init__ based on tier
        duration_days=days,
        pace=pace,
        start_time="09:00",
        end_time="18:00", # Default 6 PM
        geo_center=(30.0444, 31.2357), # Default Downtown Cairo
        indoor_preference="neutral"
    )

    print(f"\nUser Profile:")
    print(f"- Interests: {', '.join([f'{k}({v})' for k,v in user.interests.items()])}")
    print(f"- Budget Tier: {user.budget_tier} (Est. {user.budget_daily} EGP/day)")
    print(f"- Duration: {user.duration_days} days")
    print(f"- Pace: {user.pace}")
    
    # 3. Generate Itinerary
    print("\nGenerating Itinerary...")
    itinerary = sys_instance.generate_itinerary(user)
    
    # 4. Display Results
    print("\n" + "="*60)
    print("                  YOUR PERSONALIZED ITINERARY")
    print("="*60)
    
    total_trip_cost = 0
    
    for day in sorted(itinerary.keys()):
        events = itinerary[day]
        print(f"\n[ DAY {day} ]")
        if not events:
            print("  (No activities scheduled for this day - constraints too strict?)")
            continue
            
        day_cost = 0
        for event in events:
            poi = event['poi']
            print(f"  {event['start_time']} - {event['end_time']} : {poi.name}")
            print(f"    Category: {poi.category} | {poi.subcategory}")
            print(f"    Cost: {poi.cost} EGP | Duration: {poi.duration_hours}h")
            print(f"    Why: {event['reason']}")
            day_cost += poi.cost
        
        print(f"  --> Daily Total: {day_cost} EGP")
        total_trip_cost += day_cost
        print("-" * 40)
        
    print("="*60)
    print(f"TOTAL ESTIMATED TRIP COST: {total_trip_cost} EGP")
    print("="*60)

if __name__ == "__main__":
    main()
