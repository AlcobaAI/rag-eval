FROM ghcr.io/astral-sh/uv:0.11.10 AS uv_bin
FROM python:3.12-slim-bookworm AS builder

WORKDIR /app

COPY --from=uv_bin /uv /uvx /usr/local/bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-install-project --no-dev

FROM python:3.12-slim-bookworm

WORKDIR /app

# Install system essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv_bin /uv /uvx /usr/local/bin/

COPY --from=builder /app/.venv /app/.venv
COPY . .

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

CMD ["uv", "run", "deepeval", "test", "run", "tests/test_rag.py"]