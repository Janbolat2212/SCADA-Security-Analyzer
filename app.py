
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="SCADA Security Analyzer",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ SCADA Security Analyzer")
st.write("Industrial Network Traffic Security Monitoring")

st.subheader("Network Traffic Analysis")

file = st.file_uploader("Upload SCADA traffic data (CSV)", type=["csv"])

if file is not None:
    data = pd.read_csv(file)

    st.success("Dataset loaded successfully!")

    col1, col2 = st.columns(2)

    col1.metric("Total Records", len(data))
    col2.metric("Total Features", len(data.columns))

    st.subheader("Dataset Preview")
    st.dataframe(data.head(20))

    st.info("Anomaly detection will be added in the next stage.")
else:
    st.info("Please upload a CSV file to begin.")
