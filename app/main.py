"""Main application module."""

import os
import warnings
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from scripts.rag_utils import (
    load_chroma_rag,
    hybrid_search,
    ai_rerank_documents,
)
from scripts.utils import (
    load_config,
    is_ollama_running,
    does_model_exist,
    load_queries,
    init_presidio,
    clean_single_query,
    process_and_clean_queries,
)


class QueryRequest(BaseModel):
    query: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    analyzer, anonymizer = init_presidio()
    app.state.analyzer = analyzer
    app.state.anonymizer = anonymizer
    
    if not hasattr(app.state, "queries"):
        app.state.queries = []
        
    yield


app = FastAPI(title="FeedGenius", version="0.1.0", lifespan=lifespan)


@app.get("/")
async def root():
    return {"message": "Hello, World!"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/queries")
def get_queries():
    return app.state.queries


@app.post("/queries")
async def add_live_query(request: QueryRequest):
    """
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    analyzer = app.state.analyzer
    anonymizer = app.state.anonymizer

    cleaned_query = clean_single_query(request.query, analyzer, anonymizer)

    if not cleaned_query:
        raise HTTPException(status_code=400, detail="Query contains only harmful or invalid content")

    app.state.queries.append(cleaned_query)

    return {
        "status": "success",
        "original_query": request.query,
        "processed_query": cleaned_query
    }


def main():
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

    analyzer, anonymizer = init_presidio()

    query_file_path = os.getenv("QUERY_FILE_PATH", config.get("queries", {}).get("path", "/data/queries.json"))
    raw_queries = load_queries(path=query_file_path)

    if not raw_queries:
        warnings.warn(f"No queries found in the queries file at {query_file_path}")
        app.state.queries = []
    else:
        print(f"Successfully loaded {len(raw_queries)} query(ies).")
        app.state.queries = process_and_clean_queries(
            raw_queries=raw_queries, 
            analyzer=analyzer, 
            anonymizer=anonymizer
        )

    print("Application started.")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
