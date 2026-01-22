# Aurora API Dockerfile
FROM python:3.11-slim

WORKDIR /code

# System deps for xgboost
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
  && rm -rf /var/lib/apt/lists/*

# Use uv binary for faster installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install minimal deps with uv
RUN uv pip install --system fastapi uvicorn numpy pandas scikit-learn joblib xgboost

# Copy server + artifacts
COPY ./3_serve.py ./3_serve.py
COPY ./models ./models
COPY ./results ./results


EXPOSE 7860

ENTRYPOINT ["uvicorn", "3_serve:app", "--host", "0.0.0.0", "--port", "7860"]
