"""Run the API with shared application and server logging."""

import argparse

import uvicorn

from app.core.logging import logging_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run surrogate-modelling-service")
    parser.add_argument("--reload", action="store_true", help="Reload on code changes")
    args = parser.parse_args()
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        log_config=logging_config(),
        reload=args.reload,
    )
