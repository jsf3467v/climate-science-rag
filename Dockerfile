# syntax=docker/dockerfile:1
#
# Serving image for the retrieval API. CPU only, since the served stack runs no
# dense embedder. The index and chunks are large Git LFS artifacts, so
# .dockerignore keeps them out of the build and they are mounted at run time.

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app:/app/src

WORKDIR /app

# Dependencies copied before source so their layer caches across edits.
COPY requirements.txt requirements-api.txt ./
RUN pip install --upgrade pip && pip install -r requirements-api.txt

COPY . .

# Recreate the excluded mount points so a bare `docker run` still starts.
RUN mkdir -p data index

EXPOSE 8000

CMD ["uvicorn", "api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
