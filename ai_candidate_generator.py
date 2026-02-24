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
        self.poi_file = "Cairo_Giza_POI_Database_v3.xlsx"
        self.cache_file = "embeddings_cache.pkl"
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
        print(f"Loading data from {self.poi_file}...")
        if not os.path.exists(self.poi_file):
            raise FileNotFoundError(f"File not found: {self.poi_file}")
            
        self.df = pd.read_excel(self.poi_file)
        self.df.columns = [str(c).strip() for c in self.df.columns]
        
        # Ensure critical columns exist or are created
        if 'Description' not in self.df.columns:
            self.df['Description'] = self.df['Name'].astype(str) + " " + self.df['Category'].fillna('') + " " + self.df['Sub-category'].fillna('')
        
        # Parse Coordinates dynamically
        if 'Latitude' not in self.df.columns:
             # Try to find a column with "Lat" in it
            coord_col = next((c for c in self.df.columns if "Lat" in c and "Long" in c), None)
            if coord_col:
                # Helper to parse "29.9792, 31.1342"
                def parse_c(x):
                    try:
                        pts = str(x).split(',')
                        return float(pts[0]), float(pts[1])
                    except:
                        return None, None
                        
                lat_lon = self.df[coord_col].apply(parse_c)
                self.df['Latitude'] = lat_lon.apply(lambda x: x[0])
                self.df['Longitude'] = lat_lon.apply(lambda x: x[1])

        self.df['Entry cost (EGP)'] = pd.to_numeric(self.df['Entry cost (EGP)'], errors='coerce').fillna(0)
        
        # FORCE FOOD COST TO 0
        if 'Category' in self.df.columns:
             self.df.loc[self.df['Category'].astype(str).str.lower() == 'food', 'Entry cost (EGP)'] = 0

    def load_model_and_embeddings(self):
        print("Loading AI Model (SentenceTransformer)...")
        self.model = SentenceTransformer(self.model_name)
        # Load Cross-Encoder for Re-Ranking
        print("Loading Cross-Encoder (ms-marco-MiniLM-L-6-v2)...")
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Check cache validity
        cache_valid = False
        if os.path.exists(self.cache_file):
            print("Found embedding cache.")
            try:
                with open(self.cache_file, 'rb') as f:
                    data = pickle.load(f)
                    # Simple version check: compare number of rows
                    if len(data) == len(self.df):
                        self.embeddings = data
                        cache_valid = True
                        print("Cache loaded successfully.")
                    else:
                        print("Cache outdated (row count mismatch). Re-computing...")
            except (EOFError, pickle.UnpicklingError, Exception) as e:
                print(f"Cache corrupted or unreadable ({e}). Re-computing...")
                cache_valid = False
        
        if not cache_valid:
            print("Computing Embeddings (this happens once)...")
            # Create rich text representation for semantic search
            # "Name: XYZ. Category: History. Sub: Pharaonic. Context: Outdoor, 3 hours..."
            corpus = self.df.apply(lambda row: f"Name: {row['Name']}. Category: {row.get('Category','')}. Type: {row.get('Sub-category','')}. {row.get('Indoor / outdoor','')}", axis=1).tolist()
            
            self.embeddings = self.model.encode(corpus, show_progress_bar=True)
            
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.embeddings, f)
            print("Embeddings computed and cached.")

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

        # 2. Semantic Search (via Qdrant)
        print(f"📡 [AICandidateGenerator] Semantic Search for: '{query}'")
        query_vector = self.model.encode(query).tolist()
        
        try:
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k * 2
            ).points
        except:
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k * 2
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

            # Rename columns to match internal expectations
            row['Entry cost (EGP)'] = float(row.get('Entry cost (EGP)', 0))
            rows.append(row)
            
        candidates = pd.DataFrame(rows)
        if candidates.empty:
            return candidates
        
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
        top_candidates = candidates.sort_values(by='Semantic_Score', ascending=False).head(top_k * 2)
        
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
