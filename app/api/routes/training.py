"""Training transport adapters. Sync handlers run blocking work in the threadpool."""

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, Response, UploadFile
import numpy as np

from app.api.schemas.training import (TrainingConfig, TrainingErrorResponse,
                                     TrainingRunResponse, TrainingSubmission)
from app.services.data_service import DataService, DatasetTooLargeError
from app.services.training_service import TrainingRunNotFoundError, TrainingService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/training-runs", tags=["training"])
UNAVAILABLE_RESPONSE = {503: {"model": TrainingErrorResponse, "description": "Training service or backend unavailable"}}
SUBMISSION_RESPONSES = {
    **UNAVAILABLE_RESPONSE,
    202: {"headers": {"Location": {
        "description": "URL for polling the accepted training run",
        "schema": {"type": "string", "format": "uri"},
    }}},
}


def training_service(request: Request) -> TrainingService:
    service = getattr(request.app.state, "training_service", None)
    if service is None or service.is_shutdown:
        raise HTTPException(503, "Training service unavailable")
    return service


Service = Annotated[TrainingService, Depends(training_service)]


def accepted(run, response: Response, request: Request) -> TrainingRunResponse:
    response.headers["Location"] = str(request.url_for("get_training_progress", model_id=run.model_id))
    return TrainingRunResponse.model_validate(run)


def submit_error(exc: Exception) -> HTTPException:
    if isinstance(exc, DatasetTooLargeError):
        return HTTPException(413, str(exc))
    if isinstance(exc, (ValueError, TypeError)):
        return HTTPException(422, "Invalid training data or configuration")
    logger.exception("Training submission unavailable")
    return HTTPException(503, "Training service unavailable")


@router.post("", status_code=202, response_model=TrainingRunResponse, responses=SUBMISSION_RESPONSES)
def submit_json(body: TrainingSubmission, response: Response, request: Request, service: Service):
    try:
        run = service.submit(np.asarray(body.input_features), np.asarray(body.target),
                             body.input_feature_labels, body.config.to_core())
    except (ValueError, TypeError, RuntimeError) as exc:
        raise submit_error(exc) from exc
    return accepted(run, response, request)


@router.post("/upload", status_code=202, response_model=TrainingRunResponse, responses={
    **SUBMISSION_RESPONSES,
    413: {"model": TrainingErrorResponse, "description": "Compressed or expanded archive exceeds limit"},
})
def submit_upload(response: Response, request: Request, service: Service,
                  file: Annotated[UploadFile, File()],
                  config: Annotated[str | None, Form()] = None):
    try:
        settings = TrainingConfig() if config is None else TrainingConfig.model_validate_json(config)
        prepared = DataService().prepare_npz(file.file)
        run = service.submit_prepared(prepared, settings.to_core())
    except (ValueError, TypeError, RuntimeError) as exc:
        raise submit_error(exc) from exc
    finally:
        file.file.close()
    return accepted(run, response, request)


@router.get("/{model_id}", name="get_training_progress", response_model=TrainingRunResponse, responses={
    **UNAVAILABLE_RESPONSE,
    404: {"model": TrainingErrorResponse, "description": "Training run not found"},
})
def get_progress(model_id: UUID, service: Service):
    try:
        try:
            run = service.get_progress(model_id)
        except TrainingRunNotFoundError:
            run = service.get_historical_progress(model_id)
        return TrainingRunResponse.model_validate(run)
    except TrainingRunNotFoundError as exc:
        raise HTTPException(404, "Training run not found") from exc
    except Exception as exc:
        logger.exception("Training progress lookup failed for %s", model_id)
        raise HTTPException(503, "Training progress unavailable") from exc
