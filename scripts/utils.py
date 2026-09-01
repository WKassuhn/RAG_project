"""Utilities for reading config, checking Ollama server status and Query handling."""

import requests
import json
import yaml
from typing import List
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine


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


# ─── Clean Queries ────────────────────────────────────────────────────

def init_presidio() -> tuple["AnalyzerEngine", "AnonymizerEngine"]:
    """Initialisiert die Presidio Engines einmalig beim Start."""
    print("Loading Presidio NLP models...")
    return AnalyzerEngine(), AnonymizerEngine()


def clean_harmful_content(text: str) -> str:
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


def clean_single_query(query_text: str, analyzer: AnalyzerEngine, anonymizer: AnonymizerEngine) -> str:
    """
    Args:
        query_text:
        analyzer:
        anonymizer:

    Return:

    """
    if not query_text or not query_text.strip():
        return ""
        
    # 1. PII anonymisieren
    analysis_results = analyzer.analyze(text=query_text, language="en")
    anonymized_result = anonymizer.anonymize(text=query_text, analyzer_results=analysis_results)
    
    # 2. Schadcode filtern
    return clean_harmful_content(anonymized_result.text)


def process_and_clean_queries(raw_queries: list, analyzer: AnalyzerEngine, anonymizer: AnonymizerEngine) -> List[str]:
    """
    Args
        raw_queries:
        analyzer:
        anonymizer:

    Return:
        
    """
    cleaned_queries = []
    for query_item in raw_queries:
        if isinstance(query_item, dict):
            query_text = query_item.get("query") or query_item.get("text") or str(query_item)
        else:
            query_text = str(query_item)
            
        cleaned = clean_single_query(query_text, analyzer, anonymizer)
        if cleaned:
            cleaned_queries.append(cleaned)
    return cleaned_queries