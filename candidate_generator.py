import pandas as pd
import json

class CandidateGenerator:
    def __init__(self, poi_file="Cairo_Giza_500_POIs.xlsx"):
        self.preferences = {
            "group_dynamics": {},
            "interests": {},
            "travel_style": {},
            "budget": {},
            "constraints": {},
            "logistics": {}
        }
        try:
            self.df = pd.read_excel(poi_file)
            # Ensure proper types
            self.df['Entry cost (EGP)'] = pd.to_numeric(self.df['Entry cost (EGP)'], errors='coerce').fillna(0)
            self.df['Normalized_Category'] = self.df['Category'].str.lower().str.strip()
        except FileNotFoundError:
            print(f"Error: POI file '{poi_file}' not found.")
            self.df = None

    # --- Preference Collection Logic (Merged) ---
    def ask_choice(self, question, options):
        print(f"\n{question}")
        for i, opt in enumerate(options, 1):
            print(f"{i}. {opt}")
        while True:
            try:
                choice = int(input("Select an option (number): "))
                if 1 <= choice <= len(options):
                    return options[choice-1]
                print("Invalid choice.")
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

        # 1. Group Dynamics
        print("--- Step 1: Who are you traveling with? ---")
        self.preferences["group_dynamics"]["type"] = self.ask_choice("Select your group type:", ["Solo", "Couple", "Friends", "Family (with kids)", "Family (adults only)"])

        # 2. Travel Style & Pace
        print("\n--- Step 2: Travel Style & Pace ---")
        self.preferences["travel_style"]["pace"] = self.ask_choice("What is your preferred travel pace?", [
            "Relaxed (1-2 major sites/day, plenty of breaks)", 
            "Balanced (2-3 sites/day, moderate pace)", 
            "Fast-paced (See as much as possible, active)"
        ])
        
        # Energy Constraints
        self.preferences["constraints"]["walking_tolerance"] = self.ask_choice("How much walking are you comfortable with per day?", [
            "Low (< 2km)", 
            "Medium (2-5km)", 
            "High (> 5km)"
        ])

        # 3. Interests
        print("\n--- Step 3: Interests & Motivations ---")
        print("Please rate your interest in the following categories:")
        interests = ["History & Culture", "Nature & Landscapes", "Food & Dining", "Shopping", "Art & Museums", "Adventure/Sports"]
        for cat in interests:
            self.preferences["interests"][cat] = self.ask_rating(cat)

        # 4. Budget
        print("\n--- Step 4: Budget ---")
        self.preferences["budget"]["level"] = self.ask_choice("What is your budget comfort level?", [
            "Budget-conscious",
            "Standard / Mid-range",
            "Luxury / High-end"
        ])
        self.preferences["budget"]["priority"] = self.ask_choice("What would you splurge on?", ["Accommodation", "Food", "Experiences", "None"])

        # 5. Logistics
        print("\n--- Step 5: Logistics & Timing ---")
        # Transport Preference
        self.preferences["logistics"]["transport"] = self.ask_choice("Preferred mode of transport?", [
            "Private Driver / Taxi (Comfort)",
            "Public Transport / Metro (Budget/Local)",
            "Walking (where possible)",
            "Mixed"
        ])

        # Crowd Tolerance
        self.preferences["logistics"]["crowd_tolerance"] = self.ask_choice("How do you feel about crowds?", [
            "Avoid crowds (early mornings, hidden gems)",
            "Don't mind them (happy to see popular spots anytime)",
            "Prefer lively atmosphere"
        ])

        self.preferences["logistics"]["start_time"] = input("Preferred daily start time (e.g., 09:00): ")
        self.preferences["logistics"]["end_time"] = input("Preferred daily end time (e.g., 20:00): ")

        # 6. Specific Constraints
        print("\n--- Step 6: Specific Requests ---")
        must_see = input("Are there any specific places you MUST visit? (comma separated, or press Enter for none): ")
        must_avoid = input("Any places you want to AVOID? (comma separated, or Enter for none): ")
        
        if must_see.strip():
            self.preferences["constraints"]["must_visit"] = [x.strip().lower() for x in must_see.split(",")]
        else:
            self.preferences["constraints"]["must_visit"] = []
            
        if must_avoid.strip():
            self.preferences["constraints"]["must_avoid"] = [x.strip().lower() for x in must_avoid.split(",")]
        else:
            self.preferences["constraints"]["must_avoid"] = []

        return self.preferences

    # --- Candidate Generation Logic ---
    def generate_candidates(self):
        if self.df is None:
            return []

        print("\n=== Generating Candidates ===")
        
        candidates = self.df.copy()

        # 1. Hard Filter: Must Avoid
        if self.preferences["constraints"].get("must_avoid"):
            avoid_list = self.preferences["constraints"]["must_avoid"]
            # Simple substring match for avoidance
            pattern = '|'.join(avoid_list)
            candidates = candidates[~candidates['Name'].str.lower().str.contains(pattern, na=False)]

        # 2. Scoring System
        # We will assign a score to each POI based on Category match with User Interests
        
        # Map User Interest strings to Excel Categories (approximate)
        # Excel Categories seen: 'History & Culture', likely others like 'Nature', 'Shopping' etc.
        # We need to see strict category names, but for now we do partial matching.
        
        def calculate_score(row):
            score = 0
            poi_cat = str(row['Category']).lower()
            
            # Map user interest ratings to score
            # "History & Culture" -> matches 'history'
            if 'history' in poi_cat:
                score += self.preferences["interests"].get("History & Culture", 5) * 1.5 # Weight history high for Egypt
            
            if 'nature' in poi_cat or 'outdoor' in str(row['Indoor / outdoor']).lower():
                score += self.preferences["interests"].get("Nature & Landscapes", 5)
            
            if 'food' in poi_cat:
                score += self.preferences["interests"].get("Food & Dining", 5)
                
            if 'shop' in poi_cat:
                score += self.preferences["interests"].get("Shopping", 5)
                
            if 'art' in poi_cat or 'museum' in poi_cat:
                score += self.preferences["interests"].get("Art & Museums", 5)
            
            # Boost Must-Visit items significantly
            if self.preferences["constraints"].get("must_visit"):
                for wish in self.preferences["constraints"]["must_visit"]:
                    if wish in str(row['Name']).lower():
                        score += 50 # High priority boost
                        break
            
            # Budget adjustment
            cost = row['Entry cost (EGP)']
            budget_pref = self.preferences["budget"]["level"]
            
            if "Budget" in budget_pref and cost > 200:
                score -= 5 # Penalize expensive items
            elif budget_pref == "Luxury" and cost > 500:
                score += 2 # Boost premium experiences
                
            return score

        candidates['Score'] = candidates.apply(calculate_score, axis=1)
        
        # Sort by Score
        candidates = candidates.sort_values(by='Score', ascending=False)
        
        # Filter out very low scores?
        candidates = candidates[candidates['Score'] > 0]
        
        return candidates.head(50) # Return top 50 candidates

    def run(self):
        self.collect_preferences()
        results = self.generate_candidates()
        
        print("\n=== Top Recommended Candidates ===")
        if len(results) == 0:
            print("No matching candidates found.")
        else:
            print(results[['Name', 'Category', 'Entry cost (EGP)', 'Score']].to_string(index=False))
            
            # Save to a file for the next step (Itinerary Generation) to use if needed
            results.to_csv("candidates.csv", index=False)
            print("\nCandidates saved to 'candidates.csv'")

if __name__ == "__main__":
    generator = CandidateGenerator()
    generator.run()
