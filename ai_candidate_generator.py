import pandas as pd
import numpy as np
import os
import pickle
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import re

class AICandidateGenerator:
    def __init__(self, collection_name="pois", model_name='all-MiniLM-L6-v2', shared_model=None):
        self.collection_name = collection_name
        self.model_name = model_name
        self.df = None
        self.client = None
        # Removed Excel and Cache dependencies
        self.preferences = {}
        
        from qdrant_client import QdrantClient
        host = os.environ.get("QDRANT_HOST", "localhost")
        port = int(os.environ.get("QDRANT_PORT", 6333))
        self.client = QdrantClient(host=host, port=port, https=(port == 443))
        
        # Share model if provided, otherwise load fresh
        from sentence_transformers import CrossEncoder
        if shared_model:
             print("🔍 [AICandidateGenerator] Using shared Sentence Transformer.")
             self.model = shared_model
        else:
             from sentence_transformers import SentenceTransformer
             print("🔍 [AICandidateGenerator] Loading fresh Sentence Transformer…")
             self.model = SentenceTransformer(self.model_name)
             
        print("🔍 [AICandidateGenerator] Loading Cross-Encoder…")
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def load_data(self):
        """Excel loading is disabled. Data is now dynamic from Qdrant."""
        pass

    def load_model_and_embeddings(self):
        """Embedding cache is disabled. Qdrant handles vector storage."""
        print("Loading AI Model (SentenceTransformer)...")
        self.model = SentenceTransformer(self.model_name)
        # Load Cross-Encoder for Re-Ranking
        print("Loading Cross-Encoder (ms-marco-MiniLM-L-6-v2)...")
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    # --- INPUT ---
    def collect_input_interactive(self):
        # Simplified for testing AI part, utilizing same logic as before or simplified prompts
        # Ideally, we accept natural language now!
        print("\n=== AI Preference Collection ===")
        print("Describe your ideal trip in a sentence (e.g., 'I love ancient history and quiet places, but I am on a budget.')")
        self.preferences['free_text_input'] = input("> ")
        
        # We can still ask structured constraints if needed
        self.preferences['budget_max'] = float(input("Max Entry Fee (EGP) (enter 0 for no limit): ") or 0)
        
        # Geo constraint
        print("\nDo you have a location constraint? (e.g., 'Downtown Cairo', 'Giza')")
        loc_input = input("Center location (or Enter to skip): ")
        if loc_input.strip():
            # In a real app, we'd geocode this string. 
            # For now, let's look up coordinates if it matches a POI Name, or use defaults
            match = self.df[self.df['Name'].str.contains(loc_input, case=False, na=False)]
            if not match.empty:
                lat = match.iloc[0]['Latitude']
                lon = match.iloc[0]['Longitude']
                print(f"Using center: {match.iloc[0]['Name']} ({lat}, {lon})")
                self.preferences['geo_center'] = (lat, lon)
                self.preferences['geo_radius_km'] = float(input("Radius in km (e.g. 5): ") or 10)
            else:
                 print("Location not found in database, skipping geo-filter.")

    # --- API FOR MAIN SYSTEM ---
    def generate_candidates_for_user(self, user_profile, top_k=50):
        """
        Generates candidates based on a UserProfile object from the main system.
        """
        # 1. Construct Semantic Query
        # Group interests by priority buckets for cleaner sentences
        if hasattr(user_profile, 'interests') and user_profile.interests:
            sorted_interests = sorted(user_profile.interests.items(), key=lambda x: x[1], reverse=True)
            
            primary = []   # 1.0 - 0.9
            secondary = [] # 0.8 - 0.6
            tertiary = []  # 0.5 - ...
            
            for interest, weight in sorted_interests:
                if weight >= 0.9:
                    primary.append(interest)
                elif weight >= 0.6:
                    secondary.append(interest)
                else:
                    tertiary.append(interest)
            
            query_parts = []
            if primary:
                query_parts.append(f"I primarily want to visit {', '.join(primary)} places")
            if secondary:
                query_parts.append(f"I also really love {', '.join(secondary)}")
            if tertiary:
                query_parts.append(f"I am interested in {', '.join(tertiary)}")
            
            query = ". ".join(query_parts)
        else:
             query = "Popular tourist attractions in Cairo and Giza"

        # 2. Semantic Search (via Qdrant) with Hybrid Approach
        print(f"📡 [AICandidateGenerator] Semantic Search for: '{query}'")
        query_vector = self.model.encode(query).tolist()
        
        # Aggressive search depth for maximum recall (6x multiplier)
        SEARCH_DEPTH_MULTIPLIER = 6
        
        # Also generate keyword-based queries for hybrid search
        keyword_queries = []
        if interests:
            # Extract top 3 interests as keywords
            top_interests = sorted(interests.items(), key=lambda x: x[1], reverse=True)[:3]
            keyword_query = " ".join([interest for interest, _ in top_interests])
            keyword_queries.append(keyword_query)
        
        try:
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k * SEARCH_DEPTH_MULTIPLIER
            ).points
        except:
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k * SEARCH_DEPTH_MULTIPLIER
            )

        # Convert hits to DataFrame
        rows = []
        for hit in search_result:
            row = hit.payload
            row['Semantic_Score'] = hit.score
            row['id'] = hit.id
            
            # Fix: Handle Qdrant payload key for coordinates
            if 'Latitude / Longitude' in row and (row.get('Latitude') is None or row.get('Longitude') is None):
                try:
                    # Support multiple formats: "29.9, 31.1" or "29.9 / 31.1"
                    val = str(row['Latitude / Longitude'])
                    if '/' in val:
                        parts = val.split('/')
                    else:
                        parts = val.split(',')
                    
                    if len(parts) >= 2:
                        row['Latitude'] = float(parts[0].strip())
                        row['Longitude'] = float(parts[1].strip())
                except Exception as e:
                    print(f"⚠️ [AICandidateGenerator] Failed to parse coordinates '{row.get('Latitude / Longitude')}': {e}")

            # Fix: Synthesis Description if missing from payload
            if not row.get('Description'):
                name = row.get('Name', 'Unknown POI')
                cat = row.get('Category', '')
                sub = row.get('Sub-category', '')
                row['Description'] = f"{name} - {cat} - {sub}".strip(" -")

            # Keyword matching boost for hybrid search
            if interests:
                keyword_boost = 0.0
                poi_text = f"{row.get('Name', '')} {row.get('Category', '')} {row.get('Sub-category', '')} {row.get('Description', '')}".lower()
                for interest, weight in interests.items():
                    if interest.lower() in poi_text:
                        keyword_boost += weight * 0.1  # 10% boost per matching interest
                row['Keyword_Boost'] = keyword_boost
            else:
                row['Keyword_Boost'] = 0.0

            # Rename columns to match internal expectations
            row['Entry cost (EGP)'] = float(row.get('Entry cost (EGP)', 0))
            rows.append(row)
            
        candidates = pd.DataFrame(rows)
        if candidates.empty:
            return candidates
        
        # Apply hybrid scoring: Semantic Score + Keyword Boost
        if 'Keyword_Boost' in candidates.columns:
            candidates['Semantic_Score'] = candidates['Semantic_Score'] + candidates['Keyword_Boost']
        
        # Budget Filter
        # Access budget_daily safely
        budget_limit = getattr(user_profile, 'budget_daily', 10000)
        # Assuming we want individual items to be affordable within the daily budget
        # Let's say item cost shouldn't exceed 80% of daily budget? 
        # Or just filtering out insanely expensive things.
        # Actually, let's just stick to the CandidateGenerator logic:
        candidates = candidates[candidates['Entry cost (EGP)'] <= budget_limit]

        # Geo Filter (if center provided)
        geo_center = getattr(user_profile, 'geo_center', None)
        geo_radius = getattr(user_profile, 'geo_radius_km', 20.0)
        
        if geo_center:
            center_lat, center_lon = geo_center
            
            def haversine_np(lon1, lat1, lon2, lat2):
                lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
                dlon = lon2 - lon1
                dlat = lat2 - lat1
                a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
                c = 2 * np.arcsin(np.sqrt(a))
                km = 6367 * c
                return km

            # Ensure valid coords
            valid_geo_df = candidates.dropna(subset=['Latitude', 'Longitude'])
            if not valid_geo_df.empty:
                dists = haversine_np(center_lon, center_lat, valid_geo_df['Longitude'].values, valid_geo_df['Latitude'].values)
                # Assign to original index to keep alignment
                candidates.loc[valid_geo_df.index, 'Distance_km'] = dists
                # Filter
                candidates = candidates[candidates['Distance_km'] <= geo_radius]
        
        # --- RE-RANKING STEP ---
        # 1. Take top N candidates from the fast Bi-Encoder model
        #    (We take slightly more than top_k to allow re-ordering)
        # Increased to 4x for maximum recall
        top_candidates = candidates.sort_values(by='Semantic_Score', ascending=False).head(top_k * 4)
        
        if not top_candidates.empty:
            print(f"Re-ranking top {len(top_candidates)} candidates with Cross-Encoder...")
            
            # 2. Prepare Pairs: (Query, POI Description/Text)
            # Use the constructed 'query' from step 1
            # Use the 'Description' column we made in load_data, or construct on fly
            poi_texts = top_candidates['Description'].tolist()
            pairs = [[query, text] for text in poi_texts]
            
            # 3. Predict Scores
            cross_scores = self.cross_encoder.predict(pairs)
            
            # 4. Assign new scores
            top_candidates['Cross_Encoder_Score'] = cross_scores
            
            # 5. Sort by Cross-Encoder Score
            # Use this as the final semantic score
            top_candidates['Semantic_Score'] = top_candidates['Cross_Encoder_Score']
            
            # Return re-ranked
            return top_candidates.sort_values(by='Semantic_Score', ascending=False).head(top_k)
        
        return top_candidates # Fallback if empty

    # --- CLI / LEGACY METHODS ---
    def search_candidates(self):
        if self.preferences.get('free_text_input'):
            print(f"\nSemantic Searching for: '{self.preferences['free_text_input']}'...")
            query_embedding = self.model.encode([self.preferences['free_text_input']])
            
            # Cosine Similarity
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            self.df['Semantic_Score'] = similarities
        else:
            self.df['Semantic_Score'] = 0.5 # Default neutral
            
        # Filter & Rank
        candidates = self.df.copy()
        
        # 1. Budget Filter
        if self.preferences.get('budget_max', 0) > 0:
            candidates = candidates[candidates['Entry cost (EGP)'] <= self.preferences['budget_max']]
            
        # 2. Geo Filter (Vectorized Haversine)
        if 'geo_center' in self.preferences and self.preferences.get('geo_center'):
            center_lat, center_lon = self.preferences['geo_center']
            radius = self.preferences.get('geo_radius_km', 10)
            
            def haversine_np(lon1, lat1, lon2, lat2):
                lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
                dlon = lon2 - lon1
                dlat = lat2 - lat1
                a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
                c = 2 * np.arcsin(np.sqrt(a))
                km = 6367 * c
                return km

            # Ensure valid coords
            valid_geo_df = candidates.dropna(subset=['Latitude', 'Longitude'])
            if not valid_geo_df.empty:
                dists = haversine_np(center_lon, center_lat, valid_geo_df['Longitude'].values, valid_geo_df['Latitude'].values)
                valid_geo_df['Distance_km'] = dists
                # Filter
                candidates = valid_geo_df[valid_geo_df['Distance_km'] <= radius]
                print(f"Geo-filter reduced candidates to {len(candidates)} items.")
            
        # Final Sort by Semantic Match
        candidates = candidates.sort_values(by='Semantic_Score', ascending=False)
        return candidates.head(20)

if __name__ == "__main__":
    ai_gen = AICandidateGenerator()
    ai_gen.collect_input_interactive()
    results = ai_gen.search_candidates()
    
    print("\n=== AI Recommended Candidates ===")
    cols_to_show = ['Name', 'Category', 'Semantic_Score', 'Entry cost (EGP)']
    if 'Distance_km' in results.columns:
        cols_to_show.append('Distance_km')
        
    print(results[cols_to_show].to_string(index=False))
