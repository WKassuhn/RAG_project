"""Utilities for reading config, checking Ollama server status and Query handling."""

import requests
from typing import Optional
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

def load_queries():
    """
    """
    return 1

def clean_queries():
    """
    """
    import presidio_anonymizer
    import presidio_analyzer
    
    return 1

