import os
from google import genai as google_genai
from dotenv import load_dotenv

load_dotenv()

def test_gemini():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment.")
        return

    print("📡 Initializing Gemini client...")
    client = google_genai.Client(api_key=api_key)
    
    models_to_test = [
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-3-flash-preview",
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro"
    ]
    
    for model in models_to_test:
        try:
            print(f"Testing model: {model}...", end=" ")
            # Try a very simple generation
            response = client.models.generate_content(
                model=model,
                contents="test"
            )
            print("✅ SUCCESS")
        except Exception as e:
            print(f"❌ FAILED: {e}")

if __name__ == "__main__":
    test_gemini()
