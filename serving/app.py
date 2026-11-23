
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd

# Load the saved machine learning pipeline
model = joblib.load("artifacts/random_forest_model.joblib")

# Create FastAPI application
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting whether a telecom customer is likely to churn.",
    version="1.0.0"
)


# Define the input data structure
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


# Health check endpoint
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True
    }


# Prediction endpoint
@app.post("/predict")
def predict_churn(customer: CustomerData):

    # Convert input into a DataFrame
    input_data = pd.DataFrame([customer.model_dump()])

    # Generate prediction and probability
    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0][1])

    return {
        "churn_prediction": prediction,
        "churn_label": "Yes" if prediction == 1 else "No",
        "churn_probability": round(probability, 4)
    }
