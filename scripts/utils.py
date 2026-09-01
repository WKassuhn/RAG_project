"""Utilities for reading config, checking Ollama server status and Query handling."""

import requests
from typing import Optional, List, Dict
import json
import yaml
from pathlib import Path


# ─── Config ───────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)
    

# ─── Ollama ───────────────────────────────────────────────────────────

def is_ollama_running(host: str = "http://localhost:11434") -> bool:
    """
    Checks if the Ollama service is running on the specified host.
    
    Args:
        host: The base URL for the Ollama instance (e.g., 'http://localhost:11434').
                This should include the protocol and port.

    Returns:
        True if the service is reachable.

    Raises:
        RuntimeError: If the Ollama service is not reachable at the provided host.
    """
    # The /api/tags endpoint is standard for verifying Ollama is active
    url = f"{host.rstrip('/')}/api/tags"
    try:
        # Use a short timeout to avoid hanging the application during startup
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return True
        else:
            print(f"Warning: Ollama is reachable but returned status {response.status_code} at {url}")
            return True
    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            f"Ollama service not found at {host}. "
            "Please ensure Ollama is installed and running. (Error: {e})"
        ) from e
    

def does_model_exist(model_name: str = None, host: str = None) -> bool:
    """
    Checks if a specific model exists on the Ollama server.

    Args:
        model_name: The name of the model to check (e.g., 'llama3:latest').
            This can be specified directly or retrieved from configuration.

    Returns:
        True if the model is available on the server, False otherwise.
    """
    if not model_name:
        pass 

    # The /api/tags endpoint returns a list of models available on the server
    url = f"{host}/api/tags"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [m['name'] for m in data]
            return any(model_name in m for m in models) if model_name else len(models) > 0
        return False
    except Exception as e:
        print(f"Error checking model existence: {e}")
        return False


# ─── Queriy Loading ───────────────────────────────────────────────────

def load_queries(path: str = "queries.json") -> list[dict]:
    """
    Loads one or more support ticket queries from a JSON file.

    Args:
        path: The path to the JSON file containing the queries.

    Returns:
        A list of dictionaries, where each dictionary contains 
        'id', 'title', and 'description'.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            return [data] if data else []
        return data
    except FileNotFoundError:
        print(f"Warning: Query file not found at {path}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {path}: {e}")
        return []
    


def _clean_harmful_content(text: str) -> str:
    """
    Removes harmful content from a query.

    Args:
        text: The query text.

    Return:
        The processed query without harmulf content or PII.
    """
    import re

    if not text:
        return ""
        
    # Replace common prompt injection phrases
    harmful_patterns = [
        r"(ignore|override)\s+(all\s+)?(previous|prior)\s+(instructions|directives|prompts)",
        r"you\s+are\s+now\s+a\s+(bot|assistant|developer|admin)",
        r"system\s+prompt",
        r"unrestrict\s+mode"
    ]
    
    cleaned_text = text
    for pattern in harmful_patterns:
        cleaned_text = re.sub(pattern, "[REMOVED_INSTRUCTION]", cleaned_text, flags=re.IGNORECASE)
        
    cleaned_text = "".join(ch for ch in cleaned_text if ch.isprintable() or ch in ("\n", "\r", "\t"))
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text


def process_and_clean_queries(raw_queries: List[str]) -> List[str]:
    """
    """
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine

    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()
    
    cleaned_queries = []

    for query_item in raw_queries:

        query_text = ""
        if isinstance(query_item, dict):
            query_text = query_item.get("description")
        else:
            query_text = str(query_item)
            
        if not query_text.strip():
            continue
        
        analysis_results = analyzer.analyze(text=query_text, language="en")
        anonymized_result = anonymizer.anonymize(text=query_text, analyzer_results=analysis_results)
        pii_free_text = anonymized_result.text
        
        final_clean_text = _clean_harmful_content(pii_free_text)
        
        if final_clean_text.strip():
            cleaned_queries.append(final_clean_text)
            
    return cleaned_queries

