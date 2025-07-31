import os
import json
import csv
from sklearn.metrics import accuracy_score, f1_score
from datetime import datetime
import pandas as pd

# Dummy datasets and models for illustration
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Define datasets and models
DATASETS = {
    "iris": load_iris(),
    "digits": load_digits()
}

MODELS = {
    "random_forest": RandomForestClassifier(n_estimators=100),
    "logistic_regression": LogisticRegression(max_iter=1000)
}

# Output directory
RESULTS_DIR = "pipeline/results_logs"
os.makedirs(RESULTS_DIR, exist_ok=True)

CSV_PATH = os.path.join(RESULTS_DIR, "metrics_log.csv")
JSON_PATH = os.path.join(RESULTS_DIR, "metrics_log.json")

# Initialize metric logs
metrics_list = []

for dataset_name, dataset in DATASETS.items():
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.2, random_state=42)

    for model_name, model in MODELS.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        acc = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='weighted')

        entry = {
            "timestamp": datetime.now().isoformat(),
            "dataset": dataset_name,
            "model": model_name,
            "accuracy": round(acc, 4),
            "f1_score": round(f1, 4)
        }

        metrics_list.append(entry)
        print(f"Finished: {model_name} on {dataset_name} | Acc: {acc:.4f}, F1: {f1:.4f}")

# Save CSV
with open(CSV_PATH, mode='w', newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=metrics_list[0].keys())
    writer.writeheader()
    writer.writerows(metrics_list)

# Save JSON
with open(JSON_PATH, 'w') as json_file:
    json.dump(metrics_list, json_file, indent=4)

print(f"\n✅ All results saved in: {RESULTS_DIR}")
