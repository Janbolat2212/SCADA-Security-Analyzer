
import streamlit as st
import pandas as pd
import shutil
from pathlib import Path

st.set_page_config(
    page_title="SCADA Security Analyzer",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ SCADA Security Analyzer")
st.write("Industrial Network Traffic Security Monitoring")

# Folder for the SCADA dataset
DATA_FILE = Path("data/scada_dataset_V01.csv")
DATA_FILE.parent.mkdir(exist_ok=True)

st.subheader("Network Traffic Analysis")

# Upload only if the dataset is not saved yet
if not DATA_FILE.exists():
    st.info("Please upload your SCADA dataset.")

    uploaded_file = st.file_uploader(
        "Upload SCADA traffic data (CSV)",
        type=["csv"]
    )

    if uploaded_file is not None:
        with st.spinner("Saving dataset..."):
            with open(DATA_FILE, "wb") as file:
                shutil.copyfileobj(uploaded_file, file)

        st.rerun()

if DATA_FILE.exists():

    st.success("SCADA dataset loaded from saved storage!")

    # Read only the columns needed for statistics
    @st.cache_data(show_spinner="Analyzing dataset...")
    def load_data(file_path, modified_time):
        labels = pd.read_csv(
            file_path,
            usecols=["Traffic_Category", "Attack_Type"]
        )
        preview = pd.read_csv(file_path, nrows=20)
        return labels, preview

    data, preview = load_data(
        str(DATA_FILE),
        DATA_FILE.stat().st_mtime_ns
    )

    normal = (data["Traffic_Category"] == "normal_traffic").sum()
    attacks = (data["Traffic_Category"] == "attack").sum()

    st.subheader("Dataset Information")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(data))
    col2.metric("Normal Traffic", int(normal))
    col3.metric("Labeled Attacks", int(attacks))

    st.subheader("Dataset Preview")
    st.dataframe(preview)

    st.subheader("Traffic Distribution")
    st.bar_chart(data["Traffic_Category"].value_counts())

    st.subheader("Cyberattack Types")
    attack_data = data[data["Traffic_Category"] == "attack"]
    attack_counts = attack_data["Attack_Type"].value_counts()

    st.bar_chart(attack_counts)

    st.subheader("Attack Statistics")
    st.dataframe(attack_counts.rename("Records"))

    st.caption(
        "Statistics use the original dataset labels. "
        "Machine learning detection will be added next."
    )

# Machine Learning Results
st.subheader("Machine Learning Model Comparison")

rf_file = Path("results/random_forest_without_window_metrics.csv")
iso_file = Path("results/isolation_forest_metrics.csv")

if rf_file.exists() and iso_file.exists():

    rf = pd.read_csv(rf_file)
    iso = pd.read_csv(iso_file)

    comparison = pd.concat([rf, iso], ignore_index=True)

    st.write("Model Performance Metrics")
    st.dataframe(comparison.round(4))

    st.subheader("Detection Performance")

    chart_data = comparison.set_index("Model")[
        ["Precision", "Recall", "F1"]
    ]

    st.bar_chart(chart_data, stack=False)

    st.info(
        "These are preliminary experimental results. "
        "Further validation is required."
    )

else:
    st.warning("Model result files not found.")

# Unknown Attack Validation
st.subheader("Unknown Attack Detection")

validation_file = Path("results/unknown_attack_test.csv")

if validation_file.exists():

    validation = pd.read_csv(validation_file)

    st.write("Detection of attacks not seen during training")

    columns = [
        "Hidden_Attack",
        "Attacks_Tested",
        "Attacks_Detected",
        "Attacks_Missed",
        "Recall",
        "F1"
    ]

    st.dataframe(
        validation[columns].round(4),
        hide_index=True
    )

    st.subheader("Unknown Attack Detection Performance")

    chart_data = validation.set_index("Hidden_Attack")[
        ["Recall", "F1"]
    ]

    st.bar_chart(chart_data, stack=False)

    st.info(
        "Each attack type was excluded from training "
        "and tested separately."
    )

else:
    st.warning("Validation results not found.")


# Interactive Traffic Analysis
st.subheader("🔎 SCADA Traffic Analyzer")
st.caption(
    "Synthetic demonstration data. Predictions are illustrative, "
    "not measured attack-detection accuracy."
)
from joblib import load

model_file = Path("models/random_forest.joblib")
test_file = Path("demo/synthetic_modbus_demo.csv")

source = st.radio(
    "Select data source",
    ["Synthetic Demo", "Upload New CSV"],
    horizontal=True
)

uploaded = None

if source == "Upload New CSV":
    uploaded = st.file_uploader(
        "Choose a Modbus/TCP CSV file",
        type=["csv"],
        key="analysis_upload"
    )

if st.button("Analyze Traffic", type="primary"):

    if not model_file.exists():
        st.error("Trained model not found.")

    elif source == "Upload New CSV" and uploaded is None:
        st.warning("Please select a CSV file first.")

    elif source == "Synthetic Demo" and not test_file.exists():
        st.error("Test sample not found.")

    else:
        with st.spinner("Analyzing traffic..."):

            saved = load(model_file)
            features = saved["features"]

            if source == "Synthetic Demo":
                traffic = pd.read_csv(test_file)
            else:
                needed = set(features + ["Protocol"])
                traffic = pd.read_csv(
                    uploaded,
                    usecols=lambda c: c in needed
                )

            # Filter Modbus/TCP when Protocol is present
            if "Protocol" in traffic.columns:
                traffic = traffic[
                    traffic["Protocol"] == "Modbus/TCP"
                ].copy()

            missing = [
                f for f in features
                if f not in traffic.columns
            ]

            if missing:
                st.error("Missing columns: " + ", ".join(missing))

            elif traffic.empty:
                st.warning("No Modbus/TCP records found.")

            else:
                X = traffic[features].apply(
                    pd.to_numeric, errors="coerce"
                )

                predictions = saved["model"].predict(X)

                results = traffic.copy()
                results["Prediction"] = [
                    "Attack" if p == 1 else "Normal"
                    for p in predictions
                ]

                attacks = int((predictions == 1).sum())
                normal = int((predictions == 0).sum())

                st.success("Traffic analysis completed!")

                c1, c2, c3 = st.columns(3)

                c1.metric("Analyzed Records", len(results))
                c2.metric("Predicted Attacks", attacks)
                c3.metric("Predicted Normal", normal)

                st.subheader("Detection Results")
                st.bar_chart(
                    results["Prediction"].value_counts()
                )

                st.subheader("Analyzed Records")
                st.dataframe(results.head(100))

                st.download_button(
                    "Download Analysis Report",
                    results.to_csv(index=False),
                    file_name="scada_analysis_report.csv",
                    mime="text/csv"
                )

                st.caption(
                    "Experimental predictions, not verified "
                    "live industrial security alerts."
                )
