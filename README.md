# SLA Breach Monitoring - Streamlit Deployment (SAD Lab, Section B)

Predicts which student service requests may breach their SLA, with a review queue for staff.
Predictions are decision support only. Data in this repo is **synthetic**.

## Structure
app.py, requirements.txt, train_model.py, modules/ (data_processing, exploratory_analysis, prediction, dashboard),
models/trained_model.joblib, assets/use_case_diagram.png, data/sample_requests.csv

## Run locally
    pip install -r requirements.txt
    python train_model.py          # only if you want to regenerate the model
    streamlit run app.py

## Deploy
Push to GitHub -> https://share.streamlit.io -> New app -> pick repo/branch -> entry point `app.py`.

## Reuse of midterm
Replace `make_synthetic()` in train_model.py with your midterm data loading, cleaning and features,
keep the same target/features, and re-save the pipeline. Update CATEGORY_OPTIONS in modules/data_processing.py.
Note any differences from the original midterm in your deployment report.

## Limitations
Synthetic data; metrics are not real-world performance; valid only for trained categories/ranges.
