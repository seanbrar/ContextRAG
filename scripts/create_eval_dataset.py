
from pathlib import Path

import requests
import tiktoken

# Configuration
OUTPUT_DIR = Path("data/eval-mixed/documents")
TOKENIZER_NAME = "cl100k_base"

# Gutenberg URLs and metadata
# Note: Using raw text URLs. 
SOURCES = [
    {
        "title": "The Yellow Wallpaper",
        "url": "https://www.gutenberg.org/cache/epub/1952/pg1952.txt",
        "category": "medium",
        "filename": "the_yellow_wallpaper.txt"
    },
    {
        "title": "A Scandal in Bohemia",
        "url": "https://www.gutenberg.org/files/1661/1661-0.txt",
        "category": "medium",
        "filename": "a_scandal_in_bohemia.txt"
    },
    {
        "title": "The Gift of the Magi",
        "url": "https://www.gutenberg.org/files/7256/7256-0.txt",
        "category": "short", 
        "filename": "the_gift_of_the_magi.txt"
    },
    {
        "title": "The Cask of Amontillado",
        "url": "https://www.gutenberg.org/cache/epub/1063/pg1063.txt", 
        # Note: This file contains many stories, we might need to be careful or accept it's a collection.
        # Actually, let's grab a cleaner single source if possible or just use a specific known shorter text.
        # Let's stick to simple reliable URLs.
        "category": "short",
        "filename": "the_cask_of_amontillado.txt"
    },
     {
        "title": "The Gettysburg Address",
        "url": "https://www.gutenberg.org/cache/epub/4/pg4.txt",
        "category": "short", 
        "filename": "gettysburg_address.txt"
    }
]

# Manual override for The Cask of Amontillado if the full collection is too big, 
# but for now we will try to clean it. 
# actually pg1063 is "The Cask of Amontillado", likely just that story or close to it.

def count_tokens(text: str) -> int:
    encoding = tiktoken.get_encoding(TOKENIZER_NAME)
    return len(encoding.encode(text))

def clean_gutenberg_text(text: str) -> str:
    """Remove Gutenberg headers and footers."""
    # Simple heuristic: Look for start and end markers
    start_markers = ["*** START OF THE PROJECT GUTENBERG EBOOK", "*** START OF THIS PROJECT GUTENBERG EBOOK"]
    end_markers = ["*** END OF THE PROJECT GUTENBERG EBOOK", "*** END OF THIS PROJECT GUTENBERG EBOOK"]
    
    start_idx = 0
    end_idx = len(text)
    
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            # Advance past the marker line
            newline_idx = text.find("\n", idx)
            if newline_idx != -1:
                start_idx = newline_idx + 1
            break
            
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            end_idx = idx
            break
            
    return text[start_idx:end_idx].strip()

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for source in SOURCES:
        print(f"Processing {source['title']}...")
        try:
            response = requests.get(source['url'])
            response.raise_for_status()
            # Handle encoding if needed, mostly utf-8
            response.encoding = 'utf-8-sig' 
            
            raw_text = response.text
            clean_text = clean_gutenberg_text(raw_text)
            
            # Simple fallback if cleaning failed (e.g. markers mismatch) to avoid empty files
            if len(clean_text) < 100:
                print(f"Warning: Cleaning might have failed for {source['title']}, using raw text mostly.")
                # If truly failed, we might just trim a bit manually or warn.
                # Gutenberg texts are usually reliable with markers.
                if len(raw_text) > 100:
                     clean_text = raw_text # Fallback
            
            tokens = count_tokens(clean_text)
            print(f"  - Length: {len(clean_text)} chars, {tokens} tokens")
            
            output_path = OUTPUT_DIR / source['filename']
            output_path.write_text(clean_text, encoding='utf-8')
            print(f"  - Saved to {output_path}")
            
        except Exception as e:
            print(f"Error processing {source['title']}: {e}")

if __name__ == "__main__":
    main()
