# 📘 Machine Learning Model Deployment Workshop - Module 5 Update

## 🎯 Context and Aim

This workshop represents an updated version of Module 5 from the Machine Learning Zoomcamp course, originally created approximately 5 years ago. The course focuses on machine learning engineering, progressing from fundamental concepts (regression, classification, model evaluation) to advanced deployment topics. While the core theoretical concepts remain valid, this workshop modernizes the technical stack by replacing older tools with current industry-standard alternatives. The primary goal is to demonstrate the complete workflow of deploying a machine learning model as a production-ready web service, from training through containerization to cloud deployment, using contemporary tools like FastAPI, UV, and Fly.io instead of the original Flask, Pipenv, and AWS Elastic Beanstalk stack.

---

## Overview

Modernized Module 5 from ML Zoomcamp: deploying a churn prediction model as a production web service using FastAPI, UV, and Fly.io (replacing Flask, Pipenv, AWS Elastic Beanstalk).

---

## Session 1.1: Environment Setup

**Goal:** Cloud-based development environment via GitHub Codespaces

**Setup:**

- Created Codespace (Python 3.12.1, Docker pre-installed)
- Installed: `pip install jupyter scikit-learn`
- Downloaded starter notebook for churn prediction

**Key Insight:** Containerized environments eliminate "works on my machine" issues

---

## Session 1.2: Model Training & Pipelines

**Goal:** Train logistic regression model and create unified pipeline

**Core Concept:** Pipeline combines preprocessing + model into single deployable object

```python
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

pipeline = make_pipeline(
    DictVectorizer(),
    LogisticRegression(solver='liblinear')
)
pipeline.fit(train_dict, y_train)

# Save for deployment
import pickle
with open('model.bin', 'wb') as f:
    pickle.dump(pipeline, f)
```

**Key Benefits:**

- Ensures training/serving consistency
- Single artifact to serialize
- Automatic preprocessing during prediction

---

## Session 1.3: Notebook → Production Scripts

**Goal:** Convert exploratory notebook to production-ready code

**Structure:**

- `train.py`: Data loading → training → model serialization
- `predict.py`: Model loading → inference function

**Why Separate?**

- Training: Batch, resource-intensive, infrequent
- Prediction: Real-time, lightweight, continuous

**train.py essentials:**

```python
def load_data():
    # Load and preprocess data
    return df

def train_model(df):
    # Create and train pipeline
    return pipeline

def save_model(pipeline, output_file):
    # Serialize model
```

**predict.py essentials:**

```python
import pickle

with open('model.bin', 'rb') as f:
    pipeline = pickle.load(f)

# Ready for web service integration
```

---

## Key Takeaways

1. **Reproducibility:** Codespaces ensure consistent environments
2. **Pipelines:** Encapsulate entire ML workflow in one object
3. **Separation of Concerns:** Training vs. serving are different workloads
4. **Production Readiness:** Scripts > notebooks for deployment

---

## Session 3.1: Dependency Management with UV

**Goal:** Modern Python dependency isolation with UV (Rust-based, 10-100x faster than pip)

**Why UV?**

- Creates isolated virtual environments (`.venv/`)
- Lock file (`uv.lock`) ensures reproducible installations
- Follows PEP 621 standards (`pyproject.toml`)

**Setup:**

```bash
uv init
uv add scikit-learn==1.5.2 fastapi uvicorn
uv run python predict.py  # Auto-activates environment
```

**Key Files:**

- `pyproject.toml`: Human-readable dependencies
- `uv.lock`: Exact versions + cryptographic hashes (version control )
- `.venv/`: Local environment (gitignore this)

---

## Session 3.2: FastAPI Service

**Goal:** Run prediction service with proper dependency isolation

**FastAPI Implementation:**

```python
from fastapi import FastAPI
from pydantic import BaseModel

class Customer(BaseModel):
    gender: str
    tenure: int
    monthlycharges: float
    # ... (full schema with Literal types for validation)

app = FastAPI()

@app.post("/predict")
def predict(customer: Customer):
    prob = pipeline.predict_proba(customer.model_dump())[0, 1]
    return {"churn_probability": prob, "churn": prob >= 0.5}
```

**Run Service:**

```bash
uv run uvicorn predict:app --reload --host 0.0.0.0 --port 9696
# Test at http://localhost:9696/docs
```

**Key Features:**

- Pydantic validation enforces input types
- Auto-generated Swagger UI at `/docs`
- `--reload`: Auto-restart on code changes (dev only)
- `--host 0.0.0.0`: Accept external connections

---

## Session 4.1: Docker Containerization

**Goal:** Package service + dependencies in portable, isolated container

**Dockerfile Strategy:**

```dockerfile
FROM python:3.13.5-slim-bookworm

# Copy UV binary from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /code

# Enable UV-managed venv
ENV PATH="/code/.venv/bin:$PATH"

# Copy dependency files FIRST (for caching)
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked

# Copy application code LAST (changes frequently)
COPY predict.py model.bin ./

EXPOSE 9696
ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]
```

**Build & Run:**

```bash
docker build -t churn-prediction:latest .
docker run -p 9696:9696 churn-prediction:latest
```

**Why This Order?**

- Dependencies change rarely → cached layers
- Code changes frequently → only rebuilds final layer
- Faster iteration during development
- can be reloaded to update change on the fly

---

## Session 4.2: Docker Optimization

**Key Optimizations:**

1. **Layer Caching:** Dependencies before code (10x faster rebuilds)
2. **Slim Base Image:** `python:3.13.5-slim` vs full Ubuntu (smaller size)
3. **Multi-stage UV:** Copy binary from official image (no curl install)
4. **Locked Sync:** `--locked` flag prevents resolution drift
5. **`.dockerignore`:** Exclude `.git`, `.venv`, `__pycache__` (faster context)

**Security Best Practices:**

- Use specific base image tags (not `latest`)
- Minimize installed packages (slim base)
- Run as non-root user (production consideration)

### ---

## Session 5.1: Cloud Deployment (PaaS)

**Goal:** Deploy containerized service to production with public URL

**PaaS Deployment Pattern:**

```bash
# Generic workflow (works with Fly.io, Railway, Render, etc.)
1. Install platform CLI
2. Authenticate: platform auth login
3. Initialize: platform launch  # Creates config, builds, deploys
4. Access: https://your-app.platform-domain.com
5. Test: curl https://your-app.platform-domain.com/predict -d '{...}'
```

**What PaaS Handles:**

- Docker image building (from your Dockerfile)
- Container orchestration & scaling
- Global CDN/edge distribution
- Automatic HTTPS/TLS certificates
- Load balancing & health checks
- Monitoring dashboards

**What You Control:**

- Application code (predict.py)
- Dependencies (pyproject.toml)
- Container config (Dockerfile)
- Runtime environment (Python version, ports)

**PaaS vs Other Models:**

- **IaaS** (EC2, VPS): You manage OS, networking, everything → More control, more complexity
- **PaaS** (Fly.io, Render): You provide container → Platform handles infrastructure
- **Serverless** (Lambda): You provide functions → Platform handles everything but expensive at scale

**Key Testing:**

```python
# Change from localhost to production URL
url = 'https://your-app.fly.dev/predict'  # or railway.app, onrender.com
response = requests.post(url, json=customer_data)
```

**Production Checklist:**

- Remove `--reload` flag (development only)
- Monitor resource usage (RAM, CPU)
- Set up error logging/alerting  
- Consider rate limiting
- Plan for model updates (retrain → rebuild → redeploy)

---

## Complete Workflow Summary

```
1. Train Model         → train.py → model.bin
2. Build Service       → predict.py (FastAPI + Pydantic)
3. Isolate Deps        → uv (pyproject.toml + uv.lock)
4. Containerize        → Dockerfile (optimized layers)
5. Deploy to Cloud     → PaaS platform (auto-scaling)
6. Test Production     → Public URL with /docs endpoint
```

**From Notebook to Production in 6 Steps** ✅

### ---

## 🧪 Extra Tasks & Concepts

### Advanced FastAPI Features

- **Request Validation**: Pydantic models automatically validate types, ranges, and required fields, rejecting malformed requests before they reach business logic
- **Response Models**: Using `response_model` parameter ensures API contracts are enforced, preventing accidental exposure of internal data structures
- **Async Endpoints**: FastAPI supports async/await for high-concurrency workloads, enabling efficient I/O-bound operations like database queries
- **Dependency Injection**: FastAPI's dependency system enables clean code organization, request-scoped resources, and testability improvements
- **Middleware**: Custom middleware can add logging, authentication, rate limiting, and CORS policies to all endpoints systematically

### Alternative Deployment Platforms

- **AWS Elastic Beanstalk**: PaaS for AWS with deep ecosystem integration but slower deployment and more complex configuration than Fly.io
- **Google Cloud Run**: Serverless container platform that auto-scales to zero, charging only for request processing time (cost-effective for variable workloads)
- **Kubernetes**: Production-grade container orchestration for complex applications requiring advanced networking, scaling, and deployment strategies
- **AWS Lambda + API Gateway**: Serverless function deployment for simple endpoints, but cold starts and resource limits can impact ML models
- **Heroku**: Classic PaaS with extensive addon ecosystem but higher costs than alternatives for production workloads

### Model Serving Considerations

- **Latency Optimization**: Preload models at startup (not on each request), use model caching, consider model quantization for faster inference
- **Concurrent Requests**: Use asynchronous frameworks and worker processes (gunicorn) to handle multiple simultaneous predictions efficiently
- **Input Validation**: Comprehensive validation prevents adversarial inputs, data type mismatches, and invalid feature ranges from causing errors
- **Monitoring**: Implement prediction logging, latency metrics, error rates, and model performance monitoring (accuracy degradation over time)
- **Versioning**: Maintain model version metadata, enable A/B testing between model versions, and support rollback strategies

### Dependency Management Best Practices

- **Lock Files**: Always commit uv.lock to version control to ensure identical environments across team members and deployment stages
- **Security**: Regularly audit dependencies for vulnerabilities using tools like pip-audit or Snyk integration
- **Minimal Dependencies**: Reduce dependency count to minimize attack surface, reduce image size, and prevent version conflicts
- **Transitive Dependencies**: Understand that direct dependencies have their own dependencies—lock files capture complete dependency graphs
- **Development vs Production**: Separate dev dependencies (testing, linting) from production dependencies to reduce deployment image size

---

## 🧾 General Conclusion

This workshop successfully demonstrated the complete machine learning deployment lifecycle using modern tooling. The updated stack (FastAPI, UV, Docker, Fly.io) significantly improves upon the original module's tools (Flask, Pipenv, AWS EB) in terms of performance, developer experience, and deployment simplicity. Key learnings include the importance of reproducible environments through virtual environments and containers, the value of automated API documentation and validation, and the accessibility of cloud deployment platforms for making models production-ready.

**Technical Achievements:**

- Trained binary classification model achieving satisfactory validation performance for churn prediction
- Created REST API with automatic validation, documentation, and error handling using FastAPI
- Established reproducible dependency management using UV with lock file-based version control
- Containerized application for consistent deployment across development and production environments
- Deployed globally accessible prediction service to cloud platform with single-command deployment

**Production Considerations:**

- The current implementation lacks authentication, rate limiting, and comprehensive monitoring required for production systems
- Model retraining strategies and performance degradation detection should be implemented for long-term deployment
- Error handling and input validation could be enhanced to handle edge cases and malformed requests more gracefully
- Load testing should verify the service handles expected request volumes before production use
- Cost monitoring is essential when deploying to paid platforms to prevent unexpected charges
