"""
FastAPI Service for Lead Scoring Model
ML Zoomcamp 2025 - Homework 5
"""

from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# Load the pipeline
with open('pipeline_v1.bin', 'rb') as f:
    pipeline = pickle.load(f)

app = FastAPI(title="Lead Scoring API")


class LeadData(BaseModel):
    """Lead data model"""
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float


class PredictionResponse(BaseModel):
    """Prediction response model"""
    conversion_probability: float
    will_convert: bool


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Lead Scoring API",
        "status": "running",
        "model": "Logistic Regression with DictVectorizer"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(lead: LeadData):
    """
    Predict conversion probability for a lead
    
    Args:
        lead: Lead data with source, courses viewed, and income
        
    Returns:
        Conversion probability and binary prediction
    """
    # Convert to dictionary
    lead_dict = lead.dict()
    
    # Get probability
    prob = pipeline.predict_proba([lead_dict])[0, 1]
    
    # Binary prediction (threshold = 0.5)
    will_convert = prob >= 0.5
    
    return {
        "conversion_probability": float(prob),
        "will_convert": bool(will_convert)
    }


@app.post("/predict_batch")
async def predict_batch(leads: list[LeadData]):
    """
    Predict conversion probability for multiple leads
    
    Args:
        leads: List of lead data
        
    Returns:
        List of predictions
    """
    leads_dict = [lead.dict() for lead in leads]
    probs = pipeline.predict_proba(leads_dict)[:, 1]
    
    return [
        {
            "conversion_probability": float(prob),
            "will_convert": bool(prob >= 0.5)
        }
        for prob in probs
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
