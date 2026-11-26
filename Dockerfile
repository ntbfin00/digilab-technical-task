FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LOG_LEVEL=INFO

WORKDIR /service

COPY pyproject.toml ./
COPY app ./app
RUN pip install --no-cache-dir . \
    && groupadd --gid 10001 api \
    && useradd --uid 10001 --gid api --no-create-home api

USER api
EXPOSE 8000

CMD ["python", "-m", "app"]
