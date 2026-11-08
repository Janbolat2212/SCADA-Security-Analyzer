
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# Load the SCADA dataset
file_path = Path("data/scada_dataset_V01.csv")

if not file_path.exists():
    raise FileNotFoundError("SCADA dataset not found in data folder")

features = [
    "Length",
    "ip_ttl",
    "Tcp_len",
    "ip_len",
    "Frame_time_delta",
    "Tcp_hdr_len",
]

columns = ["Protocol", "Traffic_Category"] + features
data = pd.read_csv(file_path, usecols=columns, low_memory=False)

# Use only Modbus/TCP network traffic
data = data[data["Protocol"] == "Modbus/TCP"].copy()

print("SCADA Modbus/TCP records:", len(data))

# Prepare input data
X = data[features].apply(pd.to_numeric, errors="coerce")
y = (data["Traffic_Category"] == "attack").astype(int)

# Separate training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Create and train the model
model = make_pipeline(
    SimpleImputer(strategy="median"),
    RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
)

print("Training Random Forest...")
model.fit(X_train, y_train)

# Predict attacks on test data
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

# Evaluate the model
tn, fp, fn, tp = confusion_matrix(
    y_test, predictions, labels=[0, 1]
).ravel()

results = {
    "Model": "Random Forest",
    "Precision": precision_score(y_test, predictions),
    "Recall": recall_score(y_test, predictions),
    "F1": f1_score(y_test, predictions),
    "ROC_AUC": roc_auc_score(y_test, probabilities),
    "FPR": fp / (fp + tn)
}

print("\nMODEL RESULTS")
for name, value in results.items():
    print(name, ":", value)

print("\nConfusion Matrix")
print("TN:", tn, "FP:", fp)
print("FN:", fn, "TP:", tp)

# Save results for the scientific article
Path("results").mkdir(exist_ok=True)
pd.DataFrame([results]).to_csv(
    "results/random_forest_without_window_metrics.csv",
    index=False
)

print("\nResults saved successfully!")

# Save trained model
from joblib import dump

Path("models").mkdir(exist_ok=True)

dump(
    {"model": model, "features": features},
    "models/random_forest.joblib"
)

# Save test records without answer labels
X_test.to_csv(
    "data/modbus_test_sample.csv",
    index=False
)

print("Model and test data saved!")
