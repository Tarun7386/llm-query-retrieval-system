import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Try these models one by one to see which works
test_models = [
    "models/embedding-001",           # 3072 dimensions
    "models/text-embedding-004",      # 768 dimensions  
]

test_text = "What is the grace period for premium payment?"

for model_name in test_models:
    try:
        print(f"\n🔍 Testing {model_name}...")
        result = genai.embed_content(
            model=model_name,
            content=test_text,
            task_type="retrieval_document"
        )
        embedding = result["embedding"]
        print(f"✅ SUCCESS: {model_name}")
        print(f"   Dimensions: {len(embedding)}")
        print(f"   Sample values: {embedding[:3]}")
        
        # Test if embeddings are different for different texts
        result2 = genai.embed_content(
            model=model_name,
            content="Completely different text about cooking",
            task_type="retrieval_document"
        )
        embedding2 = result2["embedding"]
        
        # Calculate difference
        diff = sum(abs(a - b) for a, b in zip(embedding[:10], embedding2[:10]))
        print(f"   Difference check: {diff:.6f} (should be > 0.001)")
        
        if diff > 0.001:
            print(f"✅ {model_name} works perfectly!")
            print(f"\nUse these settings in your .env:")
            print(f"EMBEDDING_MODEL={model_name}")
            print(f"VECTOR_DIMENSION={len(embedding)}")
            break
        else:
            print(f"❌ {model_name} returns identical embeddings")
            
    except Exception as e:
        print(f"❌ ERROR with {model_name}: {e}")
