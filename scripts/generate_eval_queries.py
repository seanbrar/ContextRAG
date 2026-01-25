
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List

import openai
from dotenv import load_dotenv

# Load environment variables (for OPENAI_API_KEY)
load_dotenv()

# Configuration
DOCUMENTS_DIR = Path("data/eval-mixed/documents")
OUTPUT_FILE = Path("data/eval-mixed/queries.jsonl")
MODEL = "gpt-3.5-turbo"
QUERIES_PER_DOC = 5

def generate_queries_for_text(text: str, client: openai.OpenAI, model: str, num_queries: int = 5) -> List[str]:
    """Generate search queries for a given text using OpenAI/OpenRouter."""
    
    # Truncate text context if too long
    context = text[:15000]
    
    prompt = f"""
    You are an expert at creating evaluation datasets for retrieval systems.
    
    Here is a document:
    ---
    {context}
    ...
    ---
    
    Generate {num_queries} specific search queries that can be answered by this document.
    The queries should be diverse: some specific fact retrieval, some thematic.
    Return ONLY a JSON list of strings. Example: ["query 1", "query 2"]
    """
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates synthetic search queries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            extra_headers={
                "HTTP-Referer": "https://github.com/seanbrar/ContextRAG", # Required by OpenRouter
                "X-Title": "ContextRAG Eval Generator"
            } if "openrouter" in str(client.base_url) else None
        )
        content = response.choices[0].message.content
        # Clean up code blocks if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
            
        queries = json.loads(content.strip())
        return queries[:num_queries]
    except Exception as e:
        print(f"Error generating queries: {e}")
        return []

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = None
    model = MODEL
    
    if not api_key:
        print("OPENAI_API_KEY not found, checking for OPENROUTER_API_KEY...")
        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key:
            base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            model = os.getenv("OPENROUTER_CHAT_MODEL", "mistralai/mixtral-8x7b-instruct") # Default to a good model
            print(f"Using OpenRouter with model {model}")
        else:
            print("Error: Neither OPENAI_API_KEY nor OPENROUTER_API_KEY found.")
            return

    all_entries = []
    
    # Process each document
    files = list(DOCUMENTS_DIR.glob("*"))
    files.sort()
    
    print(f"Found {len(files)} documents. Generating queries...")
    
    # Initialize client once if possible, or per call if we prefer specific config functions
    # For simplicity, we'll pass client to the generation function or set env vars for openai lib
    # but openai lib v1+ uses client instances.
    
    client = openai.OpenAI(
        api_key=api_key,
        base_url=base_url
    )

    for file_path in files:
        if not file_path.is_file(): 
            continue
            
        print(f"Processing {file_path.name}...")
        text = file_path.read_text(encoding="utf-8")
        
        # We need to update generate_queries_for_text to accept client/model
        queries = generate_queries_for_text(text, client, model, NUM_QUERIES)
        
        doc_id = file_path.stem
        
        for q in queries:
            entry = {
                "query": q,
                "relevant_ids": [doc_id]
            }
            all_entries.append(entry)
            
        print(f"  Generated {len(queries)} queries.")
        time.sleep(1) # Rate limit politeness

        
    # Write to JSONL
    print(f"Writing {len(all_entries)} queries to {OUTPUT_FILE}...")
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for entry in all_entries:
            f.write(json.dumps(entry) + "\n")
            
    print("Done.")

if __name__ == "__main__":
    # Fix mapping for globals inside main if needed or pass as args
    NUM_QUERIES = QUERIES_PER_DOC
    main()
