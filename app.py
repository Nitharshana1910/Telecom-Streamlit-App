
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

APP_DIR = Path(__file__).resolve().parent

PREPROCESSOR_PATH = APP_DIR / "preprocessor.pkl"
XGB_MODEL_PATH = APP_DIR / "xgb_model.json"
THRESHOLD_PATH = APP_DIR / "optimal_threshold.pkl"
COLUMNS_PATH = APP_DIR / "training_columns.pkl"
SHAP_FEATURES_PATH = APP_DIR / "shap_feature_names.pkl"

st.set_page_config(
    page_title = "Telecom Churn Decision Support",
    layout = "wide"
)

# ---------------------------------------------------------------------
# Asset loading
# ---------------------------------------------------------------------

@st.cache_resource
def load_assets():
    try:
        # Load fitted preprocessing component
        preprocessor = joblib.load(
            PREPROCESSOR_PATH
        )

        # Load XGBoost using its native portable model format
        classifier = XGBClassifier()
        classifier.load_model(
            XGB_MODEL_PATH
        )

        # Reconstruct the same prediction pipeline interface used by the app
        model = Pipeline([
            ("prep", preprocessor),
            ("clf", classifier),
        ])

        optimal_threshold = float(
            joblib.load(THRESHOLD_PATH)
        )

        training_columns = list(
            joblib.load(COLUMNS_PATH)
        )

        shap_feature_names = list(
            joblib.load(SHAP_FEATURES_PATH)
        )

        return (
            model,
            optimal_threshold,
            training_columns,
            shap_feature_names,
        )

    except FileNotFoundError as error:
        st.error(
            "A required model file is missing. Make sure "
            "preprocessor.pkl, xgb_model.json, optimal_threshold.pkl, "
            "training_columns.pkl and shap_feature_names.pkl are in "
            "the same folder as app.py."
        )
        st.exception(error)
        st.stop()

    except Exception as error:
        st.error(
            "The saved model files could not be loaded."
        )
        st.exception(error)
        st.stop()


# ---------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------

def create_engineered_features(raw_customer: dict) -> pd.DataFrame:
    """Reproduce the notebook's feature engineering for one customer."""

    customer_df = pd.DataFrame([raw_customer])

    customer_df["tenure_group"] = pd.cut(
        customer_df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"],
        include_lowest=True,
    )

    services = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    customer_df["num_services"] = (
        customer_df[services] == "Yes"
    ).sum(axis=1)

    # Short-tenure guard from the final notebook:
    # for a brand-new customer, use MonthlyCharges instead of dividing by zero.
    customer_df["avg_monthly_spend"] = np.where(
        customer_df["tenure"] > 0,
        customer_df["TotalCharges"] / customer_df["tenure"],
        customer_df["MonthlyCharges"],
    )

    return customer_df


def prepare_model_input(
    raw_customer: dict,
    training_columns: list[str],
) -> pd.DataFrame:

    input_df = create_engineered_features(raw_customer)

    missing_columns = [
        column
        for column in training_columns
        if column not in input_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing model input columns: {missing_columns}"
        )

    return input_df[training_columns]


# ---------------------------------------------------------------------
# Decision support helpers
# ---------------------------------------------------------------------

def get_priority(
    probability: float,
    optimal_threshold: float,
) -> str:
    """
    Keep the priority rule aligned with the two evaluated thresholds.

    High   = also flagged by standard 0.50 threshold
    Medium = flagged only by cost-sensitive threshold
    Low    = below cost-sensitive threshold
    """
    if probability >= 0.50:
        return "High"

    if probability >= optimal_threshold:
        return "Medium"

    return "Low"


def get_recommendation(
    priority: str,
) -> str:

    if priority == "High":
        return (
            "Prioritise this customer for retention contact. "
            "Review the main risk drivers below before deciding on an offer."
        )

    if priority == "Medium":
        return (
            "Consider this customer for retention action because the "
            "cost-sensitive policy flags the customer even though the "
            "standard 0.50 rule would not."
        )

    return (
        "No immediate cost-sensitive retention action is indicated. "
        "Continue routine monitoring."
    )


def humanise_feature_name(feature_name: str) -> str:
    """Convert transformed model feature names into business-friendly labels."""

    name = feature_name.replace("cat__", "").replace("num__", "")

    replacements = {
        "Contract_Month-to-month": "Month-to-month contract",
        "Contract_One year": "One-year contract",
        "Contract_Two year": "Two-year contract",
        "OnlineSecurity_No": "No online security",
        "OnlineSecurity_Yes": "Online security enabled",
        "TechSupport_No": "No technical support",
        "TechSupport_Yes": "Technical support enabled",
        "InternetService_Fiber optic": "Fibre-optic internet",
        "InternetService_DSL": "DSL internet",
        "PaymentMethod_Electronic check": "Electronic-check payment",
        "PaymentMethod_Mailed check": "Mailed-check payment",
        "PaymentMethod_Bank transfer (automatic)": "Automatic bank transfer",
        "PaymentMethod_Credit card (automatic)": "Automatic credit-card payment",
        "PaperlessBilling_No": "No paperless billing",
        "PaperlessBilling_Yes": "Paperless billing",
        "OnlineBackup_No": "No online backup",
        "OnlineBackup_Yes": "Online backup enabled",
        "DeviceProtection_No": "No device protection",
        "DeviceProtection_Yes": "Device protection enabled",
        "MultipleLines_No": "No multiple lines",
        "MultipleLines_Yes": "Multiple lines",
        "tenure": "Customer tenure",
        "MonthlyCharges": "Monthly charges",
        "TotalCharges": "Total charges",
        "avg_monthly_spend": "Average monthly spend",
        "num_services": "Number of services",
        "SeniorCitizen": "Senior-citizen status",
    }

    return replacements.get(name, name.replace("_", " "))


# ---------------------------------------------------------------------
# SHAP helpers
# ---------------------------------------------------------------------

@st.cache_resource
def create_shap_explainer(_model_pipeline):
    classifier = _model_pipeline.named_steps["clf"]
    return shap.TreeExplainer(classifier)


def get_shap_explanation(
    _model_pipeline,
    explainer,
    input_df: pd.DataFrame,
    shap_feature_names: list[str],
):
    preprocess = _model_pipeline.named_steps["prep"]

    transformed_input = preprocess.transform(input_df)

    if hasattr(transformed_input, "toarray"):
        transformed_input = transformed_input.toarray()
    else:
        transformed_input = np.asarray(transformed_input)

    shap_values = explainer(transformed_input)

    values = np.asarray(shap_values.values[0])

    base_value = shap_values.base_values[0]
    if np.ndim(base_value) > 0:
        base_value = np.asarray(base_value).reshape(-1)[0]

    explanation = shap.Explanation(
        values=values,
        base_values=base_value,
        data=transformed_input[0],
        feature_names=shap_feature_names,
    )

    return explanation


def build_explanation_table(explanation: shap.Explanation) -> pd.DataFrame:
    shap_df = pd.DataFrame(
        {
            "Feature": explanation.feature_names,
            "SHAP Value": explanation.values,
            "Feature Value": explanation.data,
        }
    )

    shap_df["Absolute Importance"] = shap_df["SHAP Value"].abs()
    shap_df["Business Label"] = shap_df["Feature"].map(humanise_feature_name)

    # For one-hot encoded categorical features, a value of 0 means that
    # category is NOT part of this customer's profile. Those features can
    # still receive SHAP values mathematically, but showing them as plain-
    # English customer characteristics would be misleading.
    shap_df["Present in Customer Profile"] = np.where(
        shap_df["Feature"].str.startswith("cat__"),
        shap_df["Feature Value"] > 0.5,
        True,
    )

    return shap_df.sort_values(
        "Absolute Importance",
        ascending=False,
    )


def describe_feature_for_customer(
    business_label: str,
    raw_customer: dict,
) -> str:
    """Create a customer-specific, plain-English explanation."""

    mapping = {
        "Customer tenure": (
            f"Very new customer — {raw_customer['tenure']} months with the company"
            if raw_customer["tenure"] <= 3
            else f"Customer tenure — {raw_customer['tenure']} months with the company"
        ),
        "Month-to-month contract": (
            "Month-to-month contract — the customer has no long-term contract commitment"
        ),
        "One-year contract": (
            "One-year contract — the customer's contract type contributes to this individual risk estimate"
        ),
        "Two-year contract": (
            "Two-year contract — the customer's contract type contributes to this individual risk estimate"
        ),
        "Fibre-optic internet": (
            "Fibre-optic internet service — this service profile contributes to the model's risk estimate"
        ),
        "DSL internet": (
            "DSL internet service — this service profile contributes to the model's risk estimate"
        ),
        "No online security": (
            "No online security — the customer does not currently use this additional service"
        ),
        "Online security enabled": (
            "Online security enabled — this service can contribute to a lower predicted risk"
        ),
        "No technical support": (
            "No technical support — the customer does not currently use technical support"
        ),
        "Technical support enabled": (
            "Technical support enabled — this service can contribute to a lower predicted risk"
        ),
        "Electronic-check payment": (
            "Electronic-check payment — this payment method contributes to the model's risk estimate"
        ),
        "Monthly charges": (
            f"Monthly charges — {raw_customer['MonthlyCharges']:.2f} per month"
        ),
        "Total charges": (
            f"Total charges — {raw_customer['TotalCharges']:.2f} accumulated so far"
        ),
        "Average monthly spend": (
            "Average monthly spend — the customer's spending pattern contributes to the prediction"
        ),
        "No multiple lines": (
            "Single phone line — the customer does not use multiple lines"
        ),
        "Senior-citizen status": (
            "Senior-citizen status — this demographic feature contributes to the model's estimate"
        ),
        "No online backup": (
            "No online backup — the customer does not currently use this additional service"
        ),
        "No paperless billing": (
            "Paper billing — the customer is not using paperless billing"
        ),
        "Paperless billing": (
            "Paperless billing — the customer is using paperless billing"
        ),
    }

    return mapping.get(business_label, business_label)


def render_plain_english_shap(
    shap_df: pd.DataFrame,
    raw_customer: dict,
):
    # Business-facing explanations should describe characteristics that
    # actually belong to this customer. Exclude inactive one-hot categories.
    visible_df = shap_df[
        shap_df["Present in Customer Profile"]
    ].copy()

    positive = visible_df[
        visible_df["SHAP Value"] > 0
    ].head(4)

    negative = visible_df[
        visible_df["SHAP Value"] < 0
    ].head(3)

    st.markdown("#### Main reasons behind this prediction")

    if not positive.empty:
        st.markdown("**Factors pushing the predicted churn risk higher:**")
        for _, row in positive.iterrows():
            description = describe_feature_for_customer(
                row["Business Label"],
                raw_customer,
            )
            st.markdown(f"- **{description}**")

    if not negative.empty:
        st.markdown("**Factors helping reduce the predicted churn risk:**")
        for _, row in negative.iterrows():
            description = describe_feature_for_customer(
                row["Business Label"],
                raw_customer,
            )
            st.markdown(f"- **{description}**")

    st.caption(
        "Only characteristics that are present in this customer's profile "
        "are shown here. These are patterns learned by the model; they explain "
        "the prediction but do not prove that changing one feature will cause "
        "the customer to stay."
    )


# ---------------------------------------------------------------------
# Demo customer preset
# ---------------------------------------------------------------------

HIGH_RISK_CUSTOMER_PRESET = {
    "gender": "Male",
    "SeniorCitizen": 1,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 6,
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
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
    "MonthlyCharges": 95.0,
    "TotalCharges": 570.0,
}


LOW_RISK_CUSTOMER_PRESET = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "Yes",
    "tenure": 60,
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "Yes",
    "TechSupport": "Yes",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Two year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Credit card (automatic)",
    "MonthlyCharges": 55.0,
    "TotalCharges": 3300.0,
}


NEW_CUSTOMER_PRESET = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 0,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 75.0,
    "TotalCharges": 0.0,
}


RECOVERED_CHURNER_PRESET = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 24,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "One year",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 68.0,
    "TotalCharges": 1632.0,
}



def load_high_risk_customer_preset():
    for key, value in HIGH_RISK_CUSTOMER_PRESET.items():
        st.session_state[key] = value


def load_low_risk_customer_preset():
    for key, value in LOW_RISK_CUSTOMER_PRESET.items():
        st.session_state[key] = value


def load_new_customer_preset():
    for key, value in NEW_CUSTOMER_PRESET.items():
        st.session_state[key] = value


def load_recovered_churner_preset():
    for key, value in RECOVERED_CHURNER_PRESET.items():
        st.session_state[key] = value


# ---------------------------------------------------------------------
# Load final research artefacts
# ---------------------------------------------------------------------

(
    model,
    optimal_threshold,
    training_columns,
    shap_feature_names,
) = load_assets()

shap_explainer = create_shap_explainer(model)


# ---------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------

st.title("Telecom Customer Churn Decision Support System")

st.write(
    "Enter a customer profile to estimate churn risk, compare the standard "
    "and cost-sensitive decisions, receive a retention recommendation, "
    "and understand the main reasons behind the prediction."
)

top_left, top_right = st.columns([2, 1])

with top_left:
    st.info(
        "Decision policy: this system is configured to detect customers "
        "who may be at risk of leaving early, because missing a genuine "
        "churner is treated as more costly than contacting a customer who may stay."
    )

with top_right:
    if st.button(
        "Load high-risk example",
        use_container_width=True,
    ):
        load_high_risk_customer_preset()
        st.rerun()

    if st.button(
        "Load low-risk example",
        use_container_width=True,
    ):
        load_low_risk_customer_preset()
        st.rerun()

    if st.button(
        "Load new-customer example",
        use_container_width=True,
    ):
        load_new_customer_preset()
        st.rerun()


    if st.button(
        "Load cost-sensitive example",
        use_container_width=True,
        help="Example designed to show a customer that may be missed by the standard 50% rule but flagged by the earlier-retention policy.",
    ):
        load_recovered_churner_preset()
        st.rerun()


with st.expander("How does the retention decision work?"):
    st.markdown(
        f"""
The model first estimates how likely the customer is to leave.

A conventional system would usually wait until the estimated churn risk reaches
**50%** before flagging a customer.

This research system acts earlier. Customers with an estimated churn risk of
**{optimal_threshold:.0%} or more** are considered for retention action because
missing a real churner is assumed to be much more costly than contacting a
customer who eventually stays.

So **{optimal_threshold:.0%} is a decision point, not a certainty**. It does not
mean that every customer above this level will churn.
"""
    )

with st.expander("Technical details"):
    st.markdown(
        f"""
- Final model: **Tuned XGBoost**
- Cost-sensitive threshold: **{optimal_threshold:.2f}**
- Experimental FN:FP cost ratio: **10:1**
- Threshold selected using **out-of-fold training predictions**
"""
    )


# ---------------------------------------------------------------------
# Customer form
# ---------------------------------------------------------------------

with st.form("customer_form"):

    st.subheader("Customer information")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox(
            "Gender",
            ["Female", "Male"],
            key="gender",
        )

        senior_citizen = st.selectbox(
            "Senior citizen",
            [0, 1],
            format_func=lambda value: "Yes" if value == 1 else "No",
            key="SeniorCitizen",
        )

        partner = st.selectbox(
            "Partner",
            ["No", "Yes"],
            key="Partner",
        )

        dependents = st.selectbox(
            "Dependents",
            ["No", "Yes"],
            key="Dependents",
        )

        tenure = st.number_input(
            "Tenure in months",
            min_value=0,
            max_value=72,
            value=12,
            step=1,
            key="tenure",
            help=(
                "Set tenure to 0 for a customer who has just joined. "
                "The engineered spending feature uses the notebook's "
                "zero-tenure guard."
            ),
        )

        phone_service = st.selectbox(
            "Phone service",
            ["No", "Yes"],
            key="PhoneService",
        )

        multiple_lines = st.selectbox(
            "Multiple lines",
            ["No", "Yes", "No phone service"],
            key="MultipleLines",
        )

    with col2:
        internet_service = st.selectbox(
            "Internet service",
            ["DSL", "Fiber optic", "No"],
            key="InternetService",
        )

        online_security = st.selectbox(
            "Online security",
            ["No", "Yes", "No internet service"],
            key="OnlineSecurity",
        )

        online_backup = st.selectbox(
            "Online backup",
            ["No", "Yes", "No internet service"],
            key="OnlineBackup",
        )

        device_protection = st.selectbox(
            "Device protection",
            ["No", "Yes", "No internet service"],
            key="DeviceProtection",
        )

        tech_support = st.selectbox(
            "Technical support",
            ["No", "Yes", "No internet service"],
            key="TechSupport",
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["No", "Yes", "No internet service"],
            key="StreamingTV",
        )

        streaming_movies = st.selectbox(
            "Streaming movies",
            ["No", "Yes", "No internet service"],
            key="StreamingMovies",
        )

    with col3:
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"],
            key="Contract",
        )

        paperless_billing = st.selectbox(
            "Paperless billing",
            ["No", "Yes"],
            key="PaperlessBilling",
        )

        payment_method = st.selectbox(
            "Payment method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            key="PaymentMethod",
        )

        monthly_charges = st.number_input(
            "Monthly charges",
            min_value=0.0,
            value=70.0,
            step=1.0,
            key="MonthlyCharges",
        )

        total_charges = st.number_input(
            "Total charges",
            min_value=0.0,
            value=840.0,
            step=10.0,
            key="TotalCharges",
        )

    submitted = st.form_submit_button(
        "Analyse customer",
        use_container_width=True,
    )


# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------

if submitted:

    raw_customer = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    # Basic consistency warnings without blocking the prediction.
    if tenure == 0 and total_charges > monthly_charges:
        st.warning(
            "This profile represents a brand-new customer but Total Charges "
            "is relatively high. For a new-customer demonstration, Total "
            "Charges would normally be close to zero."
        )

    try:
        input_df = prepare_model_input(
            raw_customer,
            training_columns,
        )

        churn_probability = float(
            model.predict_proba(input_df)[0, 1]
        )

        standard_prediction = int(
            churn_probability >= 0.50
        )

        cost_sensitive_prediction = int(
            churn_probability >= optimal_threshold
        )

        priority = get_priority(
            churn_probability,
            optimal_threshold,
        )

        recommendation = get_recommendation(priority)

    except Exception as error:
        st.error("The prediction could not be completed.")
        st.exception(error)
        st.stop()

    st.divider()
    st.header("Customer assessment")

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric(
        "Predicted churn probability",
        f"{churn_probability:.1%}",
    )

    metric2.metric(
        "Standard 50% decision",
        "At risk" if standard_prediction else "Not at risk",
    )

    metric3.metric(
        "Recommended retention decision",
        "Take retention action" if cost_sensitive_prediction else "No immediate action",
    )

    metric4.metric(
        "Retention priority",
        priority,
    )

    st.subheader("What does this result mean?")

    if cost_sensitive_prediction and not standard_prediction:
        st.warning(
            f"The model estimates this customer's churn probability at "
            f"**{churn_probability:.1%}**. This is below the standard 50% "
            f"threshold, so a conventional classifier would not flag the "
            f"customer. However, it is above the research threshold of "
            f"**{optimal_threshold:.0%}**, so the cost-sensitive policy "
            f"recommends retention action because missing a genuine churner "
            f"is assumed to be more costly."
        )

    elif standard_prediction:
        st.error(
            f"The model estimates this customer's churn probability at "
            f"**{churn_probability:.1%}**. The customer is above both the "
            f"standard 50% threshold and the cost-sensitive "
            f"{optimal_threshold:.0%} threshold, so this is a high-priority "
            f"retention case."
        )

    else:
        st.success(
            f"The model estimates this customer's churn probability at "
            f"**{churn_probability:.1%}**. This is below the cost-sensitive "
            f"threshold of **{optimal_threshold:.0%}**, so the framework "
            f"does not currently recommend retention action."
        )

    st.subheader("Recommended action")
    st.write(recommendation)

    # -----------------------------------------------------------------
    # Explain the input profile
    # -----------------------------------------------------------------

    st.subheader("Customer profile used in the prediction")

    profile_col1, profile_col2, profile_col3 = st.columns(3)

    profile_col1.markdown(
        f"""
**Customer stage**
- Tenure: **{tenure} months**
- Contract: **{contract}**
- Partner: **{partner}**
- Dependents: **{dependents}**
"""
    )

    profile_col2.markdown(
        f"""
**Services**
- Internet: **{internet_service}**
- Online security: **{online_security}**
- Technical support: **{tech_support}**
- Online backup: **{online_backup}**
"""
    )

    profile_col3.markdown(
        f"""
**Billing**
- Monthly charges: **{monthly_charges:.2f}**
- Total charges: **{total_charges:.2f}**
- Payment method: **{payment_method}**
- Paperless billing: **{paperless_billing}**
"""
    )

    engineered_df = create_engineered_features(raw_customer)

    with st.expander("Engineered features used by the research model"):
        st.write(
            "The prototype automatically derives the same engineered "
            "features used during model development."
        )

        st.dataframe(
            engineered_df[
                [
                    "tenure_group",
                    "num_services",
                    "avg_monthly_spend",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    # -----------------------------------------------------------------
    # Standard vs cost-sensitive explanation — no technical table
    # -----------------------------------------------------------------

    st.subheader("How was the retention decision made?")

    left, right = st.columns(2)

    with left:
        st.markdown(
            f"""
### Standard decision
**Decision point: 50%**

{" This customer is flagged as a churn risk."
 if standard_prediction
 else " This customer is not flagged."}

A standard approach waits until the estimated churn risk reaches 50%.
"""
        )

    with right:
        st.markdown(
            f"""
### Cost-sensitive decision
**Decision point: {optimal_threshold:.0%}**

{" Retention action is recommended."
 if cost_sensitive_prediction
 else " No immediate retention action is recommended."}

This approach acts earlier because the research assumes that missing a genuine
churner is more costly than contacting a customer who may eventually stay.
"""
        )

    st.subheader("Suggested business response")

    if cost_sensitive_prediction:
        st.write(
            "Before contacting the customer, review contract options and "
            "available support or security services. Use the reasons below "
            "to guide the conversation rather than treating them as guaranteed causes of churn."
        )
    else:
        st.write(
            "No immediate retention intervention is recommended. Continue "
            "routine monitoring and reassess if the customer's profile changes."
        )

    # -----------------------------------------------------------------
    # SHAP explanation
    # -----------------------------------------------------------------

    st.subheader("Why is this customer at risk?")

    try:
        explanation = get_shap_explanation(
            model,
            shap_explainer,
            input_df,
            shap_feature_names,
        )

        shap_df = build_explanation_table(explanation)

        render_plain_english_shap(shap_df, raw_customer)

        with st.expander("Advanced technical explanation"):
            technical_table = shap_df.head(10).copy()

            technical_table["Effect"] = np.where(
                technical_table["SHAP Value"] > 0,
                "Increases predicted churn risk",
                "Reduces predicted churn risk",
            )

            st.dataframe(
                technical_table[
                    [
                        "Business Label",
                        "Feature Value",
                        "SHAP Value",
                        "Effect",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "This section is intended for technical reviewers. "
                "The SHAP waterfall shows how each model feature pushes the "
                "prediction toward higher or lower churn risk."
            )

            fig = plt.figure()

            shap.plots.waterfall(
                explanation,
                max_display=10,
                show=False,
            )

            st.pyplot(
                plt.gcf(),
                clear_figure=True,
            )

    except Exception as error:
        st.warning(
            "The prediction was completed, but the SHAP explanation "
            "could not be generated."
        )
        st.exception(error)