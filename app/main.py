"""Main application module."""

import os
import warnings
import uvicorn
from fastapi import FastAPI

from scripts.rag_utils import *
from scripts.utils import (
    load_config,
    is_ollama_running,
    does_model_exist,
    load_queries,
    process_and_clean_queries,
)


app = FastAPI(title="FeedGenius", version="0.1.0")


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Hello, World!"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/queries")
def get_queries():
    """"""
    return app.state.queries


def main():
    """Entry point for the application."""
    print("Initializing application...")

    config = load_config()
    base_url = config.get('ollama', {}).get('base_url', 'http://localhost:11434')
    model = config.get('llm', {}).get('model')

    try:
        is_ollama_running(host=base_url)
        print("Ollama service is reachable.")
    except RuntimeError as e:
        warnings.warn(f"Ollama service is not reachable at {base_url}. {e}")

    if model:
        try:
            if not does_model_exist(model_name=model, host=base_url):
                warnings.warn(f"Model '{model}' is not available. Please run: ollama pull {model}")
            else:
                print(f"Model '{model}' is available.")
        except Exception as e:
            warnings.warn(f"Failed to verify model '{model}': {e}")

    query_file_path = os.getenv("QUERY_FILE_PATH", config.get("queries", {}).get("path", "/data/queries.json"))
    raw_queries = load_queries(path=query_file_path)
    if not raw_queries:
        warnings.warn("No queries found in the queries file.")
    else:
        print(f"Successfully loaded {len(raw_queries)} query(ies).")

    processed_queries  = process_and_clean_queries(raw_queries=raw_queries)

    print("Application started.")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
