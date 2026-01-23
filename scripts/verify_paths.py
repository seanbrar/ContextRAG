
import sys
import os
from pathlib import Path

# Add src to sys.path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
src_path = project_root / "src"
sys.path.append(str(src_path))

from contextrag.providers.openai_chat import OpenAIChatProvider
from contextrag.providers.openrouter_chat import OpenRouterChatProvider
from contextrag.index.vector_store import VectorDB
from contextrag.config import load_config

def log_success(msg):
    print(f"✅ {msg}")

def log_failure(msg, error):
    print(f"❌ {msg}")
    print(f"   Error: {error}")

def verify_openai_chat():
    try:
        print("\n--- Verifying OpenAI Chat ---")
        provider = OpenAIChatProvider()
        response = provider.complete(
            messages=[{"role": "user", "content": "Say 'OpenAI OK'"}],
            model="gpt-3.5-turbo",
            temperature=0.0
        )
        if response:
            log_success(f"OpenAI Chat Response: {response[:50]}...")
        else:
            log_failure("OpenAI Chat returned empty response", "Empty response")
    except Exception as e:
        log_failure("OpenAI Chat Failed", e)

def verify_openrouter_chat():
    try:
        print("\n--- Verifying OpenRouter Chat ---")
        provider = OpenRouterChatProvider()
        # Using a cheap/free model for testing if possible, or the configured one
        response = provider.complete(
            messages=[{"role": "user", "content": "Say 'OpenRouter OK'"}],
            temperature=0.0
        )
        if response:
            log_success(f"OpenRouter Chat Response: {response[:50]}...")
        else:
            log_failure("OpenRouter Chat returned empty response", "Empty response")
    except Exception as e:
        log_failure("OpenRouter Chat Failed", e)

def verify_embedding_provider(provider_name, collection_name):
    try:
        print(f"\n--- Verifying {provider_name.capitalize()} Embeddings ---")
        # Use a unique collection name for testing to avoid messing up real data
        test_collection_name = f"verify_{collection_name}_{os.getpid()}"
        
        # Instantiate VectorDB with specific provider
        db = VectorDB(
            collection_name=test_collection_name, 
            embed_provider=provider_name
        )
        
        # Document to embed
        docs = ["This is a test document to verify embeddings."]
        ids = ["test_doc_1"]
        
        # Add document (triggers embedding)
        db.add_documents(documents=docs, ids=ids)
        
        # Query (triggers embedding again)
        results = db.query(query_texts=["test"], n_results=1)
        
        if results and results['documents']:
            log_success(f"{provider_name.capitalize()} Embeddings work. Query result: {results['documents'][0]}")
        else:
            log_failure(f"{provider_name.capitalize()} Embeddings return no results", "Empty results")
            
        # Cleanup (Optional: ChromaDB doesn't easily delete collections in all versions, 
        # but since we use a unique name and don't persist, it might be fine or we explicitly delete if Client allows)
        try:
            db.client.delete_collection(test_collection_name)
            print(f"   Cleaned up collection {test_collection_name}")
        except Exception as cleanup_err:
            print(f"   Warning: Could not cleanup collection {test_collection_name}: {cleanup_err}")

    except Exception as e:
        log_failure(f"{provider_name.capitalize()} Embeddings Failed", e)

def main():
    print("Starting ContextRAG Verification...")
    
    # Load config to check if env vars are set
    config = load_config()
    
    # Verify Chat
    if config.openai_api_key:
        verify_openai_chat()
    else:
        print("\n⚠️ Skipping OpenAI Chat (OPENAI_API_KEY not set)")
        
    if config.openrouter_api_key:
        verify_openrouter_chat()
    else:
        print("\n⚠️ Skipping OpenRouter Chat (OPENROUTER_API_KEY not set)")

    # Verify Embeddings
    # OpenAI
    if config.openai_api_key:
        verify_embedding_provider("openai", "openai_test")
    else:
        print("\n⚠️ Skipping OpenAI Embeddings (OPENAI_API_KEY not set)")

    # OpenRouter
    if config.openrouter_api_key:
        verify_embedding_provider("openrouter", "openrouter_test")
    else:
        print("\n⚠️ Skipping OpenRouter Embeddings (OPENROUTER_API_KEY not set)")
        
    # Local (SentenceTransformer)
    verify_embedding_provider("local", "local_test")

if __name__ == "__main__":
    main()
