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
RUN pip install --no-cache-dir xgboost
RUN mkdir -p models_log_hybrid/encoders models_log_hybrid/trained results_log_hybrid

# Copy your app code and models
COPY ./3_serve.py ./predict.py
COPY ./smoothed_target_encoder.py ./smoothed_target_encoder.py
COPY ./models_log_hybrid/encoders/target_encoder.pkl ./models_log_hybrid/encoders/target_encoder.pkl
COPY ./models_log_hybrid/trained/best_model.pkl ./models_log_hybrid/trained/best_model.pkl
COPY ./results_log_hybrid/metadata.json ./results_log_hybrid/metadata.json
COPY ./models_log_hybrid/encoders/train_medians.json ./models_log_hybrid/encoders/train_medians.json

# Expose port
EXPOSE 7860

# Run the app
ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "7860"]
