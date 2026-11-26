import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.logging import configure_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    logger.info("surrogate-modelling-service starting")
    try:
        yield
    finally:
        logger.info("surrogate-modelling-service stopped")


app = FastAPI(title="surrogate-modelling-service", lifespan=lifespan)


@app.get("/")
def health_check():
    return {"status": "running"}
