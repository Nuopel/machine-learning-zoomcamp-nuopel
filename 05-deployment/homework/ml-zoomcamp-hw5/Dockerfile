FROM agrigorev/zoomcamp-model:2025

# Set working directory
WORKDIR /code

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
# Copy your FastAPI app
COPY pyproject.toml uv.lock app_docker.py ./

# Install uv
# RUN pip install uv

# Install dependencies
RUN uv sync --frozen --no-cache

# Install dependencies from pyproject.toml
# (or just uvicorn/fastapi if pyproject.toml is missing)
# RUN pip install --no-cache-dir fastapi uvicorn scikit-learn

# Expose port
EXPOSE 8000

# Run the API
CMD ["/code/.venv/bin/uvicorn", "app_docker:app", "--host", "0.0.0.0", "--port", "8000"]