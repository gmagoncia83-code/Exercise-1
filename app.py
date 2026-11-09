import os
import pandas as pd
import streamlit as st
from modules.data_processing import load_csv, validate_schema, quality_summary, FEATURE_COLUMNS, TARGET
from modules.exploratory_analysis import render_eda
from modules.prediction import render_prediction_page
from modules.dashboard import render_dashboard

st.set_page_config(page_title="SLA Breach Monitoring - SAD", layout="wide")
PAGES = ["Home", "Dataset", "Data Analysis", "Prediction", "Dashboard", "System Design"]
menu = st.sidebar.selectbox("Module", PAGES)
st.sidebar.caption("Decision support only. Staff make the final decision.\nUse synthetic or authorized data.")
SAMPLE = "data/sample_requests.csv"


def current_df():
    df = st.session_state.get("df")
    if df is None:
        st.warning("No dataset loaded. Go to the Dataset page and upload a CSV, or load the sample data.")
    return df


st.title("Student Service Request Monitoring")

if menu == "Home":
    st.write("This system predicts which student service requests are likely to breach their "
             "service-level agreement (SLA) so student services staff can intervene early.")
    st.markdown("**Workflow:** Upload data -> Explore -> Predict -> Review risk queue -> Human decision")
    st.markdown("**Target:** `sla_breached` (1 = breach). **Prediction checkpoint:** at request submission/triage. "
                "Leakage fields such as `resolution_days` are excluded.")
    st.info("Predictions are decision-support outputs, not authorization to deny student services.")

elif menu == "Dataset":
    st.header("Dataset upload and validation")
    st.caption("Required columns: " + ", ".join(FEATURE_COLUMNS) + f" (optional: {TARGET}, request_id)")
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if st.button("Use sample synthetic data") and os.path.exists(SAMPLE):
        st.session_state["df"] = pd.read_csv(SAMPLE)
    if uploaded is not None:
        df, err = load_csv(uploaded)
        if err:
            st.error(err)
        else:
            ok, msgs = validate_schema(df)
            for m in msgs:
                (st.info if m.startswith("Note:") else st.error)(m)
            if ok:
                st.session_state["df"] = df
                st.success("Dataset passed validation.")
    df = st.session_state.get("df")
    if df is not None:
        q = quality_summary(df)
        a, b, c, d = st.columns(4)
        a.metric("Rows", q["rows"]); b.metric("Columns", q["columns"])
        c.metric("Duplicate rows", q["duplicate_rows"]); d.metric("Missing values", q["total_nulls"])
        st.dataframe(df.head(20), use_container_width=True)
        if q["total_nulls"]:
            st.dataframe(q["null_table"])

elif menu == "Data Analysis":
    st.header("Exploratory data analysis")
    df = current_df()
    if df is not None:
        render_eda(df)

elif menu == "Prediction":
    st.header("Prediction")
    render_prediction_page()

elif menu == "Dashboard":
    st.header("Decision-support dashboard")
    df = current_df()
    if df is not None:
        render_dashboard(df)

else:
    st.header("System design (SAD integration)")
    st.subheader("Actors")
    st.write("- **Student Services Officer** - uploads data, views predictions, records final decisions\n"
             "- **Registrar / Unit Head** - views dashboard and review queue, oversees decisions\n"
             "- **System Admin** - manages datasets and model versions")
    st.subheader("Functional requirements")
    st.write("1. FR1: Upload a CSV of requests and validate schema, nulls and duplicates.\n"
             "2. FR2: Show descriptive statistics, class distribution and interactive charts.\n"
             "3. FR3: Predict SLA-breach probability for a new request using the saved model.\n"
             "4. FR4: Show KPIs and a risk-ranked review queue with recommended follow-up.\n"
             "5. FR5: Let an authorized officer record the final decision with reviewer and timestamp.")
    st.subheader("Non-functional requirements")
    st.write("1. NFR1 Performance: pages respond within about 3 seconds (model loaded once, cached).\n"
             "2. NFR2 Privacy/security: only synthetic or authorized data; no credentials in the repository.\n"
             "3. NFR3 Transparency: model version, timestamp, limitations and human review are visible.")
    st.subheader("Updated use case diagram")
    st.image("assets/use_case_diagram.png")
    st.subheader("Decision workflow")
    st.graphviz_chart("""
    digraph G { rankdir=LR; node [shape=box, style=rounded];
      A [label="Upload / receive\\nrequest data"]; B [label="Validate schema"];
      C [label="Model predicts\\nbreach probability"]; D [label="Risk tier +\\nreview queue"];
      E [label="Officer reviews", shape=diamond]; F [label="Record decision\\n(escalate / monitor / none)"];
      G [label="Follow-up action\\nand audit log"];
      A -> B -> C -> D -> E -> F -> G; }
    """)
    st.caption("Only authorized staff may view results and record final decisions; the model never decides alone.")
