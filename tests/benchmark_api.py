
import httpx
import time
import numpy as np
import csv
from pathlib import Path

# API address
API_URL = "http://127.0.0.1:8000/predict"

# Number of requests
NUM_REQUESTS = 100
WARMUP_REQUESTS = 10

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

latencies = []

with httpx.Client(timeout=30.0) as client:

    # Warm-up requests (not included in measurements)
    for _ in range(WARMUP_REQUESTS):
        response = client.post(API_URL, json=sample_customer)
        response.raise_for_status()

    print("Warm-up completed.")
    print("Starting API benchmark...")

    # Measure API response time
    start_time = time.perf_counter()

    for i in range(NUM_REQUESTS):

        request_start = time.perf_counter()

        response = client.post(
            API_URL,
            json=sample_customer
        )

        request_end = time.perf_counter()

        response.raise_for_status()

        latency = (request_end - request_start) * 1000
        latencies.append(latency)

        print(f"Request {i + 1}: {latency:.2f} ms")

    total_time = time.perf_counter() - start_time


# Calculate performance metrics
p50 = np.percentile(latencies, 50)
p95 = np.percentile(latencies, 95)
p99 = np.percentile(latencies, 99)

average_latency = np.mean(latencies)
throughput = NUM_REQUESTS / total_time

print("\n----- API BENCHMARK RESULTS -----")
print(f"Total requests: {NUM_REQUESTS}")
print(f"Average latency: {average_latency:.2f} ms")
print(f"p50 latency: {p50:.2f} ms")
print(f"p95 latency: {p95:.2f} ms")
print(f"p99 latency: {p99:.2f} ms")
print(f"Throughput: {throughput:.2f} requests/second")


# Save individual request measurements
reports_dir = Path("reports")
reports_dir.mkdir(exist_ok=True)

with open(reports_dir / "api_request_latencies.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Request Number", "Latency (ms)"])

    for i, latency in enumerate(latencies, start=1):
        writer.writerow([i, round(latency, 2)])


# Save benchmark summary
with open(reports_dir / "api_benchmark_summary.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Total Requests",
        "Average Latency (ms)",
        "P50 (ms)",
        "P95 (ms)",
        "P99 (ms)",
        "Throughput (requests/sec)"
    ])

    writer.writerow([
        NUM_REQUESTS,
        round(average_latency, 2),
        round(p50, 2),
        round(p95, 2),
        round(p99, 2),
        round(throughput, 2)
    ])

print("\nBenchmark results saved in reports folder.")
