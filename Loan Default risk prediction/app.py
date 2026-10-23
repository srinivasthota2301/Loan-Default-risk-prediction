import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, roc_auc_score)

st.set_page_config(page_title="Loan Default Risk Predictor", layout="wide")

FEATURES = ["income", "loan_amount", "credit_score", "credit_history_years",
            "employment_years", "existing_loans", "loan_to_income"]


# ---------------------------------------------------------------- DATA
@st.cache_data
def make_data(n=5000, seed=42):
    """Synthetic loan data. Replace with pd.read_csv('your_file.csv') for real data."""
    rng = np.random.default_rng(seed)
    income = rng.lognormal(mean=13.3, sigma=0.5, size=n)            # annual income (Rs)
    loan_amount = income * rng.uniform(0.1, 3.0, n)
    credit_score = np.clip(rng.normal(680, 90, n), 300, 900)
    credit_history_years = np.clip(rng.normal(7, 4, n), 0, 30)
    employment_years = np.clip(rng.normal(6, 4, n), 0, 35)
    existing_loans = rng.poisson(1.5, n)
    lti = loan_amount / income

    logit = (-1.8 + 1.1 * lti - 0.012 * (credit_score - 650)
             - 0.08 * credit_history_years - 0.05 * employment_years
             + 0.35 * existing_loans + rng.normal(0, 0.5, n))
    default = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    return pd.DataFrame({
        "income": income, "loan_amount": loan_amount, "credit_score": credit_score,
        "credit_history_years": credit_history_years,
        "employment_years": employment_years, "existing_loans": existing_loans,
        "loan_to_income": lti, "default": default})


# --------------------------------------------------------------- MODELS
@st.cache_resource
def train_models():
    df = make_data()
    X, y = df[FEATURES], df["default"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight="balanced",
            random_state=42, n_jobs=-1),
    }
    results = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        proba = m.predict_proba(X_test)[:, 1]
        results[name] = {
            "model": m, "proba": proba,
            "cm": confusion_matrix(y_test, pred),
            "metrics": {
                "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred),
                "Recall": recall_score(y_test, pred),
                "F1-score": f1_score(y_test, pred),
                "ROC-AUC": roc_auc_score(y_test, proba)}}
    return results, y_test, df


def risk_category(p):
    if p < 0.30:
        return "Low Risk", "green"
    if p < 0.60:
        return "Medium Risk", "orange"
    return "High Risk", "red"


results, y_test, df = train_models()

# ------------------------------------------------------------------- UI
st.title("💳 Loan Default Risk Predictor")
st.caption("Logistic Regression & Random Forest | Streamlit + Scikit-learn")

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Model Performance", "🗂 Data"])

# ---- Tab 1: prediction
with tab1:
    model_name = st.sidebar.selectbox("Model", list(results.keys()))
    st.sidebar.markdown("**Risk bands**\n- Low: < 30%\n- Medium: 30–60%\n- High: > 60%")

    c1, c2 = st.columns(2)
    with c1:
        income = st.number_input("Annual income (₹)", 100000, 50000000, 600000, step=50000)
        loan_amount = st.number_input("Loan amount (₹)", 10000, 100000000, 500000, step=50000)
        existing_loans = st.number_input("Existing loans", 0, 15, 1)
    with c2:
        credit_score = st.slider("Credit score (300–900)", 300, 900, 700)
        credit_history_years = st.slider("Credit history (years)", 0, 30, 5)
        employment_years = st.slider("Employment (years)", 0, 35, 4)

    if st.button("Predict Risk", type="primary"):
        row = pd.DataFrame([[income, loan_amount, credit_score, credit_history_years,
                             employment_years, existing_loans, loan_amount / income]],
                           columns=FEATURES)
        p = results[model_name]["model"].predict_proba(row)[0, 1]
        label, color = risk_category(p)
        st.markdown(f"### Result: :{color}[{label}]")
        st.metric("Default probability", f"{p:.1%}")
        st.progress(float(p))

        fig, ax = plt.subplots(figsize=(6, 1.2))
        ax.barh([0], [0.30], color="green", alpha=.6)
        ax.barh([0], [0.30], left=0.30, color="orange", alpha=.6)
        ax.barh([0], [0.40], left=0.60, color="red", alpha=.6)
        ax.axvline(p, color="black", lw=3)
        ax.set_xlim(0, 1); ax.set_yticks([]); ax.set_xlabel("Default probability")
        st.pyplot(fig)

# ---- Tab 2: performance
with tab2:
    st.subheader("Metrics comparison (20% test set)")
    metrics_df = pd.DataFrame({k: v["metrics"] for k, v in results.items()}).T
    st.dataframe(metrics_df.style.format("{:.3f}"), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        metrics_df.drop(columns="ROC-AUC").plot(kind="bar", ax=ax)
        ax.set_ylim(0, 1); ax.set_title("Model comparison"); plt.xticks(rotation=0)
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        for name, r in results.items():
            fpr, tpr, _ = roc_curve(y_test, r["proba"])
            ax.plot(fpr, tpr, label=f"{name} (AUC={r['metrics']['ROC-AUC']:.2f})")
        ax.plot([0, 1], [0, 1], "k--"); ax.legend()
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC curve")
        st.pyplot(fig)

    col3, col4 = st.columns(2)
    with col3:
        sel = st.selectbox("Confusion matrix for", list(results.keys()), key="cm")
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(results[sel]["cm"],
                               display_labels=["No Default", "Default"]).plot(ax=ax, cmap="Blues")
        ax.set_title(f"Confusion Matrix - {sel}")
        st.pyplot(fig)
    with col4:
        rf = results["Random Forest"]["model"]
        imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
        fig, ax = plt.subplots(figsize=(5, 4))
        imp.plot(kind="barh", ax=ax, color="teal")
        ax.set_title("Feature importance (Random Forest)")
        st.pyplot(fig)

# ---- Tab 3: data
with tab3:
    st.write(f"Rows: {len(df)} | Default rate: {df['default'].mean():.1%}")
    st.dataframe(df.head(50), use_container_width=True)