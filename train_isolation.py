
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# Load SCADA data
file_path = Path("data/scada_dataset_V01.csv")

features = [
    "Length",
    "ip_ttl",
    "Tcp_len",
    "ip_len",
    "Frame_time_delta",
    "Tcp_hdr_len"
]

columns = ["Protocol", "Traffic_Category"] + features

data = pd.read_csv(
    file_path,
    usecols=columns,
    low_memory=False
)

# Select Modbus/TCP traffic
data = data[data["Protocol"] == "Modbus/TCP"].copy()

X = data[features].apply(pd.to_numeric, errors="coerce")
y = (data["Traffic_Category"] == "attack").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Train only on normal traffic
normal_train = X_train[y_train == 0]

model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
        n_jobs=-1
    )
)

print("Training Isolation Forest...")
model.fit(normal_train)

# Isolation Forest: -1 = anomaly, 1 = normal
predictions = (model.predict(X_test) == -1).astype(int)
anomaly_scores = -model.decision_function(X_test)

tn, fp, fn, tp = confusion_matrix(
    y_test, predictions, labels=[0, 1]
).ravel()

results = {
    "Model": "Isolation Forest",
    "Precision": precision_score(
        y_test, predictions, zero_division=0
    ),
    "Recall": recall_score(y_test, predictions),
    "F1": f1_score(y_test, predictions),
    "ROC_AUC": roc_auc_score(y_test, anomaly_scores),
    "FPR": fp / (fp + tn)
}

print("\nMODEL RESULTS")

for name, value in results.items():
    print(name, ":", value)

print("\nConfusion Matrix")
print("TN:", tn, "FP:", fp)
print("FN:", fn, "TP:", tp)

Path("results").mkdir(exist_ok=True)

pd.DataFrame([results]).to_csv(
    "results/isolation_forest_metrics.csv",
    index=False
)

print("\nResults saved successfully!")
