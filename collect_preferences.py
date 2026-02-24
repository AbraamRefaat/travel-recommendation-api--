import json

class UserPreferenceCollector:
    def __init__(self):
        self.preferences = {
            "group_dynamics": {},
            "interests": {},
            "travel_style": {},
            "budget": {},
            "constraints": {},
            "logistics": {}
        }

    def ask_choice(self, question, options):
        print(f"\n{question}")
        for i, opt in enumerate(options, 1):
            print(f"{i}. {opt}")
        while True:
            try:
                choice = int(input("Select an option (number): "))
                if 1 <= choice <= len(options):
                    return options[choice-1]
                print("Invalid choice, please try again.")
            except ValueError:
                print("Please enter a number.")

    def ask_rating(self, category):
        while True:
            try:
                rating = float(input(f"Rate your interest in {category} (0-10): "))
                if 0 <= rating <= 10:
                    return rating
                print("Please enter a number between 0 and 10.")
            except ValueError:
                print("Invalid input.")

    def collect_preferences(self):
        print("=== Candidate Generation Preference Collector ===")
        print("Please answer the following questions to help us tailor your itinerary.\n")

        # 1. Group Dynamics (Page 6)
        print("--- Step 1: Who are you traveling with? ---")
        group_type = self.ask_choice("Select your group type:", ["Solo", "Couple", "Friends", "Family (with kids)", "Family (adults only)"])
        self.preferences["group_dynamics"]["type"] = group_type
        
        # 2. Travel Style & Pace (Page 2, 4)
        print("\n--- Step 2: Travel Style & Pace ---")
        pace = self.ask_choice("What is your preferred travel pace?", [
            "Relaxed", 
            "Moderate", 
            "Packed"
        ])
        self.preferences["travel_style"]["pace"] = pace
        
        # Energy Constraints
        walking_tol = self.ask_choice("How much walking are you comfortable with per day?", [
            "Low (< 2km)", 
            "Medium (2-5km)", 
            "High (> 5km)"
        ])
        self.preferences["constraints"]["walking_tolerance"] = walking_tol

        # 3. Interests (Page 3)
        print("\n--- Step 3: Interests & Motivations ---")
        print("Please rate your interest in the following categories:")
        interests = ["History", "Food", "Nature", "Shopping", "Entertainment", "Religious"]
        for cat in interests:
            val = self.ask_rating(cat)
            self.preferences["interests"][cat] = val
            
        # 4. Budget (Page 4)
        print("\n--- Step 4: Budget ---")
        budget_level = self.ask_choice("What is your budget comfort level?", [
            "Budget",
            "Moderate",
            "Luxury"
        ])
        self.preferences["budget"]["level"] = budget_level
        self.preferences["budget"]["priority"] = self.ask_choice("What would you splurge on?", ["Accommodation", "Food", "Experiences", "None"])

        # 5. Logistics (Page 5)
        print("\n--- Step 5: Logistics & Timing ---")
        
        # Transport Preference
        transport = self.ask_choice("Preferred mode of transport?", [
            "Private Driver / Taxi (Comfort)",
            "Public Transport / Metro (Budget/Local)",
            "Walking (where possible)",
            "Mixed"
        ])
        self.preferences["logistics"]["transport"] = transport

        # Crowd Tolerance (Page 7)
        crowds = self.ask_choice("How do you feel about crowds?", [
            "Avoid crowds (early mornings, hidden gems)",
            "Don't mind them (happy to see popular spots anytime)",
            "Prefer lively atmosphere"
        ])
        self.preferences["logistics"]["crowd_tolerance"] = crowds

        start_time = input("Preferred daily start time (e.g., 09:00): ")
        end_time = input("Preferred daily end time (e.g., 20:00): ")
        self.preferences["logistics"]["start_time"] = start_time
        self.preferences["logistics"]["end_time"] = end_time

        # 6. Specific Constraints
        print("\n--- Step 6: Specific Requests ---")
        must_see = input("Are there any specific places you MUST visit? (comma separated, or press Enter for none): ")
        must_avoid = input("Any places you want to AVOID? (comma separated, or Enter for none): ")
        
        if must_see.strip():
            self.preferences["constraints"]["must_visit"] = [x.strip() for x in must_see.split(",")]
        if must_avoid.strip():
            self.preferences["constraints"]["must_avoid"] = [x.strip() for x in must_avoid.split(",")]

        print("\n=== Preference Collection Complete ===")
        return self.preferences

if __name__ == "__main__":
    collector = UserPreferenceCollector()
    data = collector.collect_preferences()
    print("\nCollected User Preferences:")
    print(json.dumps(data, indent=4))
