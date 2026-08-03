FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# lxml / trafilatura need a compiler toolchain for some wheels
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libxml2-dev libxslt1-dev curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY artemis ./artemis
RUN pip install --upgrade pip && pip install .

RUN useradd --create-home --uid 10001 artemis \
    && mkdir -p /app/.artemis-cache \
    && chown -R artemis:artemis /app
USER artemis

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "artemis.main:app", "--host", "0.0.0.0", "--port", "8000"]
