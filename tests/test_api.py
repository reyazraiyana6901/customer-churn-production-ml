
from fastapi.testclient import TestClient
from serving.app import app

client = TestClient(app)

# Sample customer data
sample_customer = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.7,
    "TotalCharges": 171.4
}


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_prediction_endpoint():
    response = client.post("/predict", json=sample_customer)

    assert response.status_code == 200

    result = response.json()

    assert result["churn_prediction"] in [0, 1]
    assert result["churn_label"] in ["Yes", "No"]
    assert 0 <= result["churn_probability"] <= 1


def test_invalid_input():
    invalid_customer = sample_customer.copy()
    invalid_customer["SeniorCitizen"] = 5

    response = client.post("/predict", json=invalid_customer)

    assert response.status_code == 422
