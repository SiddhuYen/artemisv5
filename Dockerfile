FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# curl is for the healthcheck. Deliberately NOT libxml2-dev/libxslt1-dev and a
# compiler: with those present pip will build lxml from source against the
# system libxml2, and that build segfaulted this container under concurrent
# parsing — `double free or corruption (out)`, exit 139. lxml ships manylinux
# wheels with a tested libxml2 statically bundled, so the toolchain bought a
# riskier build of something we could have had prebuilt.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY artemis ./artemis
# --only-binary lxml makes the source fallback a build failure rather than a
# silent segfault three runs later.
RUN pip install --upgrade pip \
    && pip install --only-binary lxml,lxml_html_clean .

RUN useradd --create-home --uid 10001 artemis \
    && mkdir -p /app/.artemis-cache \
    && chown -R artemis:artemis /app
USER artemis

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "artemis.main:app", "--host", "0.0.0.0", "--port", "8000"]
