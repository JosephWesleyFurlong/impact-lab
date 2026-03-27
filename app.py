import streamlit as st
import pandas as pd
import numpy as np
from dowhy import CausalModel
import os
from graphviz import Digraph


st.write("API Key Loaded:", bool(os.getenv("OPENAI_API_KEY")))
st.set_page_config(page_title="Causal Inference MVP", layout="wide")

st.title("🧠 Causal Inference MVP for Program Evaluation")

# -----------------------------
# Synthetic Data Generator
# -----------------------------S
@st.cache_data
def generate_data(n=1000, seed=42):
    np.random.seed(seed)

    age = np.random.randint(5, 18, size=n)
    trauma = np.random.normal(50, 10, size=n)
    prior_placements = np.random.poisson(2, size=n)

    # Treatment assignment (confounded)
    mentoring_prob = 1 / (1 + np.exp(-(0.05*trauma + 0.3*prior_placements)))
    mentoring = np.random.binomial(1, mentoring_prob)

    # Outcome (true causal effect included)
    disruption_prob = 1 / (1 + np.exp(
        -(0.04*trauma + 0.5*prior_placements - 0.8*mentoring)
    ))
    disruption = np.random.binomial(1, disruption_prob)

    df = pd.DataFrame({
        "age": age,
        "trauma_score": trauma,
        "prior_placements": prior_placements,
        "mentoring": mentoring,
        "disruption": disruption
    })

    return df


def suggest_variables(df):
    columns = df.columns.tolist()

    treatment = None
    outcome = None
    confounders = []

    for col in columns:
        if any(x in col.lower() for x in ["treat", "program", "mentoring", "intervention"]):
            treatment = col
            break

    for col in columns:
        if any(x in col.lower() for x in ["outcome", "result", "disruption", "success"]):
            outcome = col
            break

    if not treatment:
        treatment = columns[-2]

    if not outcome:
        outcome = columns[-1]

    confounders = [c for c in columns if c not in [treatment, outcome]]

    return treatment, outcome, confounders

from openai import OpenAI
import json

client = OpenAI()

def ai_suggest_variables(question, columns):
    prompt = f"""
You are helping build a causal inference model.

Dataset columns:
{columns}

User question:
"{question}"

Return ONLY valid JSON with:
- treatment (must be one of the columns)
- outcome (must be one of the columns)
- confounders (list of columns)

Rules:
- Only use column names from the dataset
- Choose the most likely causal interpretation
- Confounders should influence both treatment and outcome
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a causal inference expert."},
            {"role": "user", "content": prompt}
        ]
    )

    try:
        result = json.loads(response.choices[0].message.content)
        return result
    except:
        return None

def explain_method_agreement(question, treatment, outcome, confounders, psm, reg):
    prompt = f"""
You are helping interpret causal inference results.

User question:
"{question}"

Model:
- Treatment: {treatment}
- Outcome: {outcome}
- Confounders: {confounders}

Results:
- Propensity Score Matching (PSM): {psm}
- Linear Regression: {reg}

Explain:
1. Whether the methods agree or differ
2. Why regression might show a smaller effect than PSM
3. What this means for interpreting the results

Important:
- Differences do NOT necessarily mean one method is wrong
- Highlight concepts like heterogeneity, overlap, and model assumptions
- Keep it clear and practical for a non-technical audience
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

def explain_causal_model(question, treatment, outcome, confounders):
    prompt = f"""
You are helping explain a causal inference model in plain language.

User question:
"{question}"

Model:
- Treatment: {treatment}
- Outcome: {outcome}
- Confounders: {confounders}

Explain:
1. Why this treatment and outcome were chosen
2. Why the confounders matter
3. What assumptions are being made

Keep it simple and clear for a non-technical audience.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

def explain_method_differences(question, treatment, outcome, confounders, psm, reg):
    prompt = f"""
You are helping explain differences between causal inference methods.

User question:
"{question}"

Model:
- Treatment: {treatment}
- Outcome: {outcome}
- Confounders: {confounders}

Results:
- Propensity Score Matching: {psm}
- Linear Regression: {reg}

Explain:
1. Why these methods might give different results
2. What assumptions differ
3. When to trust each method

Keep it simple and practical for a non-technical audience.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


# -----------------------------
# Data Section
# -----------------------------
st.sidebar.header("1. Data")

data_option = st.sidebar.radio(
    "Choose data source:",
    ["Use synthetic data", "Upload CSV"]
)

if data_option == "Use synthetic data":
    df = generate_data()
    st.sidebar.success("Synthetic dataset loaded")
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        st.stop()

st.subheader("Preview Data")
st.dataframe(df.head())

# -----------------------------
# Optional AI Assistant
# -----------------------------
st.divider()
st.subheader("💬 Ask Your Evaluation Question (Optional)")

st.caption("Type a question and press Enter or click submit to get help selecting variables.")

# Initialize session state if needed
if "treatment" not in st.session_state:
    st.session_state.treatment = None
    st.session_state.outcome = None
    st.session_state.confounders = []

# Use a form so Enter submits properly
with st.form("question_form"):
    question = st.text_input(
        "Example: Did mentoring reduce placement disruptions?"
    )

    submitted = st.form_submit_button("✨ Suggest Model")

    if submitted:
        if not question:
            st.warning("Please enter a question or use manual selection below.")
        else:
            st.info(f"Interpreting question: '{question}'")

            try:
                ai_result = ai_suggest_variables(question, df.columns.tolist())
            except Exception as e:
                st.warning("AI service error. Falling back to default.")
                ai_result = None

            if ai_result:
                t = ai_result.get("treatment")
                o = ai_result.get("outcome")
                c = ai_result.get("confounders", [])

                # ✅ Validate AI output
                valid_columns = df.columns.tolist()
                if t not in valid_columns or o not in valid_columns:
                    st.warning("AI returned invalid columns. Using fallback.")
                    ai_result = None
                else:
                    st.session_state.treatment = t
                    st.session_state.outcome = o
                    st.session_state.confounders = c

                    st.success("AI suggested a model. You can adjust below.")

                    # 🧠 Explanation
                    explanation = explain_causal_model(question, t, o, c)

                    st.subheader("🧠 Why this model?")
                    st.write(explanation)

                    st.json(ai_result)

            if not ai_result:
                st.warning("Using default variable selection.")

                t, o, c = suggest_variables(df)

                st.session_state.treatment = t
                st.session_state.outcome = o
                st.session_state.confounders = c

                st.success("Default model applied. You can adjust below.")


# -----------------------------
# Variable Selection
# -----------------------------
st.sidebar.header("2. Define Variables")

columns = df.columns.tolist()

treatment = st.selectbox(
    "Treatment Variable",
    columns,
    index=columns.index(st.session_state.treatment)
    if st.session_state.treatment in columns else 0
)

outcome = st.selectbox(
    "Outcome Variable",
    columns,
    index=columns.index(st.session_state.outcome)
    if st.session_state.outcome in columns else 1
)

confounders = st.multiselect(
    "Confounders",
    [c for c in columns if c not in [treatment, outcome]],
    default=[c for c in st.session_state.confounders if c not in [treatment, outcome]]
)


st.session_state.treatment = treatment
st.session_state.outcome = outcome
st.session_state.confounders = confounders

# -----------------------------
# Build DAG
# -----------------------------
def build_graph(treatment, outcome, confounders):
    graph = "digraph {\n"
    graph += f"{treatment} -> {outcome};\n"

    for c in confounders:
        graph += f"{c} -> {treatment};\n"
        graph += f"{c} -> {outcome};\n"

    graph += "}"
    return graph

graph = build_graph(treatment, outcome, confounders)

st.subheader("Causal Graph (DAG)")

dot = Digraph()

# Style nodes
dot.attr('node', shape='box', style='filled', color='lightblue')

# Treatment node (highlight)
dot.node(treatment, treatment, color='lightgreen')

# Outcome node (highlight)
dot.node(outcome, outcome, color='lightcoral')

# Edges
dot.edge(treatment, outcome)

for c in confounders:
    dot.node(c, c, color='lightblue')
    dot.edge(c, treatment)
    dot.edge(c, outcome)

st.graphviz_chart(dot)

if treatment == outcome:
    st.error("Treatment and outcome must be different.")
    st.stop()

if treatment in confounders or outcome in confounders:
    st.warning("Treatment or outcome should not also be listed as confounders.")

# -----------------------------
# Run Analysis
# -----------------------------
st.sidebar.header("3. Run Analysis")

if st.sidebar.button("Estimate Effect"):

    try:
        # -----------------------------
        # Data Diagnostics
        # -----------------------------
        st.subheader("🔍 Data Diagnostics")

        treatment_counts = df[treatment].value_counts()

        st.write("Treatment Distribution:")

        treatment_counts_df = (
            treatment_counts
            .rename_axis(treatment)
            .reset_index(name="count")
        )

        st.dataframe(treatment_counts_df)

        # -----------------------------
        # Trust Indicators (NEW)
        # -----------------------------
        st.subheader("🧭 Data Quality Signals")

        min_group = treatment_counts.min()

        if min_group > 100:
            st.success("🟢 Strong treatment balance — results are more reliable")
        elif min_group > 30:
            st.warning("🟡 Moderate imbalance — interpret results with caution")
        else:
            st.error("🔴 Severe imbalance — results may be unreliable")

        # -----------------------------
        # Guardrails (existing)
        # -----------------------------
        if treatment_counts.min() < 10:
            st.warning("Very small group detected. Estimates may be unstable.")

        if treatment_counts.nunique() < 2:
            st.error("Treatment must have at least two groups (e.g., 0 and 1).")
            st.stop()

        
                # -----------------------------
        # Build model
        # -----------------------------
        model = CausalModel(
            data=df,
            treatment=treatment,
            outcome=outcome,
            graph=graph
        )

        identified_estimand = model.identify_effect()

        estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.propensity_score_matching"
        )

        # -----------------------------
        # Alternative Method: Regression (NEW)
        # -----------------------------
        regression_estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.linear_regression"
        )

        # -----------------------------
        # Results (Multi-Method)
        # -----------------------------
        st.subheader("📊 Results (Multi-Method Comparison)")

        psm_effect = estimate.value
        reg_effect = regression_estimate.value
        # Try to extract additional statistics safely
        # -----------------------------
# Diagnostics Table (prep)
# -----------------------------
        psm_se = getattr(estimate, "stderr", None)
        reg_se = getattr(regression_estimate, "stderr", None)

        diagnostics_df = pd.DataFrame({
            "Method": ["Propensity Score Matching", "Linear Regression"],
            "Effect": [psm_effect, reg_effect],
            "Std Error": [psm_se, reg_se],
            "N (approx)": [len(df), len(df)]
        })

        if reg_se is not None:
            diagnostics_df["CI Lower"] = diagnostics_df["Effect"] - 1.96 * diagnostics_df["Std Error"]
            diagnostics_df["CI Upper"] = diagnostics_df["Effect"] + 1.96 * diagnostics_df["Std Error"]


        results_df = pd.DataFrame({
            "Method": ["Propensity Score Matching", "Linear Regression"],
            "Estimated Effect": [psm_effect, reg_effect]
        })

        st.dataframe(results_df)

        st.subheader("🧠 Method Comparison")

        diff = abs(psm_effect - reg_effect)

        if diff < 0.05:
            st.success("🟢 Methods are consistent — this strengthens confidence in the result.")

        elif diff < 0.15:
            st.warning("🟡 Methods differ somewhat — results may depend on modeling assumptions.")

        else:
            st.info("""
        🔍 Methods differ substantially.

        This does not necessarily mean one is wrong. Differences can arise due to:
        - Treatment effects varying across individuals
        - Limited overlap between groups
        - Model assumptions (e.g., linearity in regression)

        Consider examining subgroup effects or overlap.
        """)

        

        # -----------------------------
        # Practical Interpretation
        # -----------------------------
        st.subheader("🎯 Practical Interpretation")

        # Baseline rate
        baseline = df[outcome].mean() * 100

        # Convert effects to percentage points
        psm_pct = psm_effect * 100
        reg_pct = reg_effect * 100

        st.write(
            f"The baseline rate of {outcome} is approximately {baseline:.1f}%."
        )

        st.write(
            f"The estimated effect ranges from {psm_pct:.1f}% to {reg_pct:.1f}% (percentage points)."
        )

        # Directional interpretation
        if psm_effect < 0:
            st.success(
                f"📉 Mentoring may reduce the likelihood of disruption by about "
                f"{abs(psm_pct):.1f}%."
            )
        else:
            st.warning(
                f"📈 Mentoring may increase the likelihood of disruption by about "
                f"{abs(psm_pct):.1f}%."
            )

        # Intuitive framing
        st.write(
            f"In practical terms, this means about {abs(psm_pct):.0f} fewer disruptions "
            f"per 100 similar cases."
        )
        

        # -----------------------------
        # Interpretation
        # -----------------------------
        st.subheader("🧾 Interpretation")

        effect = estimate.value

        st.write(
            f"On average, units that received the treatment had an outcome "
            f"{'lower' if effect < 0 else 'higher'} by approximately {abs(effect):.3f}."
        )

        st.write("""
        ⚠️ This result depends on:
        - Correct model specification  
        - Inclusion of all relevant confounders  
        - No hidden bias  
        """)

        # -----------------------------
        # Refutation Test
        # -----------------------------
        st.subheader("🔍 Refutation Test")

        refutation = model.refute_estimate(
            identified_estimand,
            estimate,
            method_name="placebo_treatment_refuter"
        )

        st.write(refutation)

                # -----------------------------
        # Confidence Summary (NEW)
        # -----------------------------
        st.subheader("🧭 Overall Confidence")

        signals = []

        # 1. Balance signal
        if min_group > 100:
            signals.append("good_balance")
        elif min_group > 30:
            signals.append("moderate_balance")
        else:
            signals.append("poor_balance")

        # 2. Method agreement
        diff = abs(psm_effect - reg_effect)
        if diff < 0.05:
            signals.append("strong_agreement")
        elif diff < 0.15:
            signals.append("moderate_agreement")
        else:
            signals.append("weak_agreement")

        # 3. Refutation signal (simple heuristic)
        ref_text = str(refutation)
        if "significant" in ref_text.lower():
            signals.append("refutation_pass")
        else:
            signals.append("refutation_unclear")


        # -----------------------------
        # Model Diagnostics (TOGGLE)
        # -----------------------------
        with st.expander("📊 Model Diagnostics (click to expand)", expanded=False):

            st.write("Detailed model output for technical review:")

            st.dataframe(diagnostics_df)

            st.caption("""
            Notes:
            - Standard errors may vary depending on method assumptions
            - PSM focuses on matched samples
            - Regression uses all available data
            """)
        # -----------------------------
        # Interpret signals
        # -----------------------------
        if "poor_balance" in signals or "weak_agreement" in signals:
            st.error("🔴 Low confidence in results")
        elif "moderate_balance" in signals or "moderate_agreement" in signals:
            st.warning("🟡 Moderate confidence — interpret with caution")
        else:
            st.success("🟢 High confidence in results")

        # Optional: show details
        # -----------------------------
        # -----------------------------
        # AI Explanation of Agreement (TOGGLE)
        # -----------------------------
        with st.expander(
            "🧠 Interpreting Method Differences (click to expand)",
            expanded=False  # ✅ ensures it's closed by default
        ):
            try:
                explanation = explain_method_agreement(
                    question if 'question' in locals() else "",
                    treatment,
                    outcome,
                    confounders,
                    psm_effect,
                    reg_effect
                )
                st.write(explanation)
            except Exception:
                st.warning("Could not generate explanation.")

        # -----------------------------
        # Export Report (NEW)
        # -----------------------------
        # -----------------------------
        # -----------------------------
        # Export Report
        # -----------------------------
        st.subheader("📄 Export Report")

        report = f"""
CAUSAL ANALYSIS REPORT
----------------------

Question:
{question if 'question' in locals() else 'Not provided'}

Treatment:
{treatment}

Outcome:
{outcome}

Confounders:
{', '.join(confounders)}

--------------------------------------
DATA DIAGNOSTICS
--------------------------------------
{treatment_counts.to_string()}

--------------------------------------
RESULTS
--------------------------------------
Estimated Treatment Effect (ATE):
{estimate.value:.4f}

--------------------------------------
INTERPRETATION
--------------------------------------
On average, the treatment {'reduced' if estimate.value < 0 else 'increased'} the outcome.

--------------------------------------
LIMITATIONS
--------------------------------------
- Observational data (not randomized)
- Assumes no unmeasured confounders
- Results depend on model specification
"""

        st.download_button(
            label="⬇️ Download Report",
            data=report,
            file_name="causal_analysis_report.txt",
            mime="text/plain"
        )

    except Exception as e:
        st.error(f"Error: {e}")