FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LOG_LEVEL=INFO

WORKDIR /service

COPY pyproject.toml ./
COPY app ./app
# Install CPU-only PyTorch first so the application install reuses it.
RUN pip install --no-cache-dir "torch>=2.6,<3" \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir ".[dev]" \
    && groupadd --gid 10001 api \
    && useradd --uid 10001 --gid api --no-create-home api

USER api
EXPOSE 8000

CMD ["python", "-m", "app"]
