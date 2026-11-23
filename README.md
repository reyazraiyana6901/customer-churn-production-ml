
# Customer Churn Prediction – ML Model Engineering

## 1. Project Overview

This project develops a machine learning system to predict whether a telecom customer is likely to leave the company (churn).

The project covers data preprocessing, exploratory data analysis, model development, evaluation, model serialization and deployment through a FastAPI application.

## 2. Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- FastAPI
- Pydantic
- Joblib
- Pytest

## 3. Project Structure

(Add final project folder structure here.)

## 4. Machine Learning Models

- Logistic Regression
- Class-Weighted Logistic Regression
- Random Forest

## 5. Final Model Results

(Add final test-set evaluation results here.)

## 6. API Development

The trained model is deployed using FastAPI.

Available endpoints:

- GET /health
- POST /predict

Interactive API documentation is available at /docs when the application is running.

## 7. API Performance

The API was benchmarked using 100 sequential requests.

(Add measured latency and throughput results here.)

## 8. Drift Monitoring

A monitoring strategy is proposed to track changes in incoming customer data, prediction behaviour and API performance.

## 9. How to Run the Project

(Add setup and execution instructions here.)

## 10. Limitations

(Add project limitations here.)

## 11. Dataset and References

(Add dataset source and relevant references here.)
ytest cache directory #

This directory contains data from the pytest's cache plugin,
which provides the `--lf` and `--ff` options, as well as the `cache` fixture.

**Do not** commit this to version control.

See [the docs](https://docs.pytest.org/en/stable/how-to/cache.html) for more information.
