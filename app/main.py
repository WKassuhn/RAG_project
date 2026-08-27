"""Main application module."""

from fastapi import FastAPI
from scripts.rag_utils import *
from scripts.utils import *


app = FastAPI(title="My App", version="0.1.0")


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Hello, World!"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


def main():
    """Entry point for the application."""
    import uvicorn

    print("Application started.")
    uvicorn.run(app, host="0.0.0.0", port=8000)

    config = load_config()

    base_url = config.get('ollama', {}).get('base_url')
    ollama_status = is_ollama_running(host=base_url)

    model = config.get('llm', {}).get('model')
    model_status = does_model_exist(model_name=model, host=base_url)

if __name__ == "__main__":
    main()
