FROM ghcr.io/astral-sh/uv:0.11.5-python3.12-alpine AS builder

WORKDIR /app

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-install-project

FROM python:3.12.3-alpine3.19

WORKDIR /app

RUN apk add --no-cache libstdc++ g++

COPY --from=builder /app/.venv /app/.venv

COPY . .

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

CMD ["deepeval", "test", "run", "tests/test_rag.py"]