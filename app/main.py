import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app.api.routes.training import router as training_router
from app.core.logging import configure_logging
from app.services.training_service import TrainingService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    logger.info("surrogate-modelling-service starting")
    app.state.training_service = None
    project = os.getenv("WANDB_PROJECT", "").strip()
    entity = os.getenv("WANDB_ENTITY", "").strip()
    if project and entity:
        app.state.training_service = TrainingService(project=project, entity=entity)
    service = app.state.training_service
    try:
        yield
    finally:
        app.state.training_service = None
        if service is not None:
            await run_in_threadpool(service.shutdown)
        logger.info("surrogate-modelling-service stopped")


app = FastAPI(title="surrogate-modelling-service", lifespan=lifespan)
app.include_router(training_router)


@app.exception_handler(RequestValidationError)
async def invalid_request(request, exc: RequestValidationError):
    # Raw invalid inputs can contain NaN/Infinity, which JSONResponse cannot
    # serialize. Retain useful locations/messages without echoing those values.
    return JSONResponse(status_code=422, content={"detail": [
        {key: error[key] for key in ("type", "loc", "msg")}
        for error in exc.errors()
    ]})


@app.get("/")
def health_check():
    return {"status": "running"}
