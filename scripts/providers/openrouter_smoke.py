#!/usr/bin/env python3
"""
OpenRouter API Test Script

Tests the OpenRouter API with structured outputs using a free model.
Validates that the API key works and structured responses are returned correctly.
"""

import json
import os
import sys

import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Free model for testing (supports structured outputs)
MODEL = "mistralai/devstral-2512:free"


def test_structured_output():
    """Test OpenRouter API with structured outputs."""
    
    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY not found in environment variables.")
        print("Make sure .env file exists with OPENROUTER_API_KEY=...")
        sys.exit(1)
    
    print(f"Testing OpenRouter API with model: {MODEL}")
    print("-" * 50)
    
    # Define the structured output schema
    schema = {
        "type": "object",
        "properties": {
            "scientist_name": {
                "type": "string",
                "description": "Full name of the scientist"
            },
            "field": {
                "type": "string",
                "description": "Primary field of study"
            },
            "major_contribution": {
                "type": "string",
                "description": "One major scientific contribution"
            },
            "birth_year": {
                "type": "integer",
                "description": "Year of birth"
            }
        },
        "required": ["scientist_name", "field", "major_contribution", "birth_year"],
        "additionalProperties": False
    }
    
    # Make the API request
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": "Give me information about Marie Curie in the requested format."
            }
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "scientist_info",
                "strict": True,
                "schema": schema
            }
        }
    }
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    
    print("Sending request to OpenRouter API...")
    print()
    
    try:
        response = requests.post(
            OPENROUTER_API_URL,
            headers=headers,
            json=payload,
            timeout=60
        )
        
        # Check for HTTP errors
        if response.status_code != 200:
            print(f"ERROR: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            sys.exit(1)
        
        data = response.json()
        
        # Check for API errors
        if "error" in data:
            print(f"API Error: {data['error']}")
            sys.exit(1)
        
        # Extract and parse the structured response
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        
        print("SUCCESS! Received structured response:")
        print("-" * 50)
        print(json.dumps(parsed, indent=2))
        print("-" * 50)
        
        # Validate the response structure
        required_fields = ["scientist_name", "field", "major_contribution", "birth_year"]
        missing = [f for f in required_fields if f not in parsed]
        
        if missing:
            print(f"WARNING: Missing expected fields: {missing}")
        else:
            print("All required fields present in response.")
        
        # Print usage info if available
        if "usage" in data:
            usage = data["usage"]
            print()
            print(f"Token usage: {usage.get('prompt_tokens', '?')} prompt, "
                  f"{usage.get('completion_tokens', '?')} completion, "
                  f"{usage.get('total_tokens', '?')} total")
        
        print()
        print("OpenRouter integration test PASSED!")
        return True
        
    except requests.exceptions.Timeout:
        print("ERROR: Request timed out")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Request failed: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse response as JSON: {e}")
        print(f"Raw content: {content}")
        sys.exit(1)
    except KeyError as e:
        print(f"ERROR: Unexpected response structure, missing key: {e}")
        print(f"Full response: {json.dumps(data, indent=2)}")
        sys.exit(1)


if __name__ == "__main__":
    test_structured_output()
