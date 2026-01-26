"""Verification script for ContextRAG providers and embeddings."""

import os
import sys
from pathlib import Path

# Add src to sys.path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
src_path = project_root / "src"
sys.path.append(str(src_path))

from chromaroute import VectorStore

from contextrag.config import load_config
from contextrag.embeddings.provider import build_embedding_function
from contextrag.providers.openai_chat import OpenAIChatProvider
from contextrag.providers.openrouter_chat import OpenRouterChatProvider


def log_success(msg: str) -> None:
    print(f"✅ {msg}")


def log_failure(msg: str, error: Exception | str) -> None:
    print(f"❌ {msg}")
    print(f"   Error: {error}")


def verify_openai_chat() -> None:
    try:
        print("\n--- Verifying OpenAI Chat ---")
        provider = OpenAIChatProvider()
        response = provider.complete(
            messages=[{"role": "user", "content": "Say 'OpenAI OK'"}],
            model="gpt-3.5-turbo",
            temperature=0.0,
        )
        if response:
            log_success(f"OpenAI Chat Response: {response[:50]}...")
        else:
            log_failure("OpenAI Chat returned empty response", "Empty response")
    except Exception as e:
        log_failure("OpenAI Chat Failed", e)


def verify_openrouter_chat() -> None:
    try:
        print("\n--- Verifying OpenRouter Chat ---")
        provider = OpenRouterChatProvider()
        response = provider.complete(
            messages=[{"role": "user", "content": "Say 'OpenRouter OK'"}],
            temperature=0.0,
        )
        if response:
            log_success(f"OpenRouter Chat Response: {response[:50]}...")
        else:
            log_failure("OpenRouter Chat returned empty response", "Empty response")
    except Exception as e:
        log_failure("OpenRouter Chat Failed", e)


def verify_embedding_provider(provider_name: str, collection_name: str) -> None:
    try:
        print(f"\n--- Verifying {provider_name.capitalize()} Embeddings ---")
        test_collection_name = f"verify_{collection_name}_{os.getpid()}"

        config = load_config()
        embedding_fn = build_embedding_function(
            config=config,
            embed_provider=provider_name,
        )
        store = VectorStore(
            collection_name=test_collection_name,
            embedding_function=embedding_fn,
        )

        docs = ["This is a test document to verify embeddings."]
        ids = ["test_doc_1"]

        store.add_documents(documents=docs, ids=ids)
        results = store.query(query_texts=["test"], n_results=1)

        if results and results["documents"]:
            log_success(
                f"{provider_name.capitalize()} Embeddings work. "
                f"Query result: {results['documents'][0]}"
            )
        else:
            log_failure(
                f"{provider_name.capitalize()} Embeddings return no results",
                "Empty results",
            )

        try:
            store.delete_collection()
            print(f"   Cleaned up collection {test_collection_name}")
        except Exception as cleanup_err:
            print(
                f"   Warning: Could not cleanup collection "
                f"{test_collection_name}: {cleanup_err}"
            )

    except Exception as e:
        log_failure(f"{provider_name.capitalize()} Embeddings Failed", e)


def main() -> None:
    print("Starting ContextRAG Verification...")

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
    if config.openrouter_api_key:
        verify_embedding_provider("openrouter", "openrouter_test")
    else:
        print("\n⚠️ Skipping OpenRouter Embeddings (OPENROUTER_API_KEY not set)")

    # Local (SentenceTransformer)
    verify_embedding_provider("local", "local_test")


if __name__ == "__main__":
    main()
