"""Main application module."""

from fastapi import FastAPI

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


if __name__ == "__main__":
    main()
