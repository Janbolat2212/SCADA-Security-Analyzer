
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# Load Modbus/TCP records
file = Path("data/scada_dataset_V01.csv")

features = [
    "Length", "ip_ttl", "Tcp_len",
    "ip_len", "Frame_time_delta", "Tcp_hdr_len"
]

columns = ["Protocol", "Traffic_Category", "Attack_Type"] + features

data = pd.read_csv(file, usecols=columns, low_memory=False)
data = data[data["Protocol"] == "Modbus/TCP"].copy()
data[features] = data[features].apply(
    pd.to_numeric, errors="coerce"
)

# Split normal traffic
normal = data[data["Traffic_Category"] == "normal_traffic"]
normal_train, normal_test = train_test_split(
    normal, test_size=0.25, random_state=42
)

results = []
attack_types = sorted(
    data.loc[data["Traffic_Category"] == "attack", "Attack_Type"].unique()
)

for hidden_attack in attack_types:

    # Exclude one attack type from training
    known_attacks = data[
        (data["Traffic_Category"] == "attack") &
        (data["Attack_Type"] != hidden_attack)
    ]

    unknown_attacks = data[
        (data["Traffic_Category"] == "attack") &
        (data["Attack_Type"] == hidden_attack)
    ]

    training = pd.concat([normal_train, known_attacks])
    testing = pd.concat([normal_test, unknown_attacks])

    y_train = (training["Traffic_Category"] == "attack").astype(int)
    y_test = (testing["Traffic_Category"] == "attack").astype(int)

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

    print("\nTesting unknown attack:", hidden_attack)

    model.fit(training[features], y_train)

    predictions = model.predict(testing[features])
    probabilities = model.predict_proba(testing[features])[:, 1]

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    result = {
        "Hidden_Attack": hidden_attack,
        "Attacks_Tested": len(unknown_attacks),
        "Attacks_Detected": int(tp),
        "Attacks_Missed": int(fn),
        "False_Alarms": int(fp),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1": f1_score(y_test, predictions, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, probabilities),
        "FPR": fp / (fp + tn)
    }

    results.append(result)

    print("Detected:", tp, "out of", tp + fn)
    print("Recall:", round(result["Recall"] * 100, 2), "%")
    print("F1:", round(result["F1"] * 100, 2), "%")

# Save experiment results
Path("results").mkdir(exist_ok=True)

pd.DataFrame(results).to_csv(
    "results/unknown_attack_test.csv",
    index=False
)

print("\nValidation completed. Results saved!")
