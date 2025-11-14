# Use official slim Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /code

# Copy only requirements first for caching
COPY pyproject.toml ./

# Install pip-tools and build environment
# Upgrade pip and install project dependencies from pyproject.toml
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
# RUN pip install --no-cache-dir xgboost --only-binary :all:
RUN pip install --no-cache-dir .

# Copy your app code and models
COPY ./3_serve.py ./predict.py
COPY ./target_encoder.py ./target_encoder.py
COPY ./models/encoders/target_encoder.pkl ./models/encoders/target_encoder.pkl
COPY ./models/trained/LinearRegression_te.pkl ./models/trained/LinearRegression_te.pkl

# Expose port
EXPOSE 7860

# Run the app
ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "7860"]
