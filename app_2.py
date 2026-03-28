import json, os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import streamlit as st
from dowhy import CausalModel
from graphviz import Digraph

st.set_page_config(page_title="Causal Inference MVP", layout="wide")

# ── Session state ──────────────────────────────────────────────────────────────
for k, v in dict(treatment=None, outcome=None, confounders=[], df_clean=None,
                 analysis_ran=False, use_did=False, psm_effect=None,
                 reg_effect=None, did_effect=None, did_se=None,
                 min_group=None, question="", dataset_explanation=None,
                 quiz_done=False, quiz_result=None).items():
    st.session_state.setdefault(k, v)

# ── OpenAI helper ──────────────────────────────────────────────────────────────
try:
    from openai import OpenAI
    _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    _AI = True
except Exception:
    _client = None; _AI = False

def _chat(prompt, system="You are a causal inference expert helping program directors who are not statisticians."):
    if not _AI: return None
    try:
        r = _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": prompt}])
        return r.choices[0].message.content
    except Exception as e:
        st.warning(f"AI error: {e}"); return None

# ── Data generators ────────────────────────────────────────────────────────────
@st.cache_data
def generate_backdoor_data(n=1000, seed=42):
    rng = np.random.default_rng(seed)
    trauma = rng.normal(50, 10, n)
    prior  = rng.poisson(2, n)
    ment   = rng.binomial(1, 1/(1+np.exp(-(0.05*trauma+0.3*prior))))
    disrupt= rng.binomial(1, 1/(1+np.exp(-(0.04*trauma+0.5*prior-0.8*ment))))
    return pd.DataFrame({"age": rng.integers(5,18,n), "trauma_score": trauma,
                         "prior_placements": prior, "mentoring": ment, "disruption": disrupt})

@st.cache_data
def generate_did_data(n=1000, seed=42):
    """
    Realistic DiD dataset: youth mentoring program evaluated across 10 quarters.
    - program_enrolled: 1 if the youth was enrolled in the mentoring program, 0 if comparison group
    - quarter: observation period (1-10); program launched at quarter 6 (post=1)
    - post_launch: 1 for quarters 6-10, 0 for quarters 1-5
    - age, prior_placements: background variables (confounders)
    - placement_disruptions: outcome — number of placement disruptions that quarter
    """
    rng = np.random.default_rng(seed)
    n_subjects = n // 10
    subject_ids = np.repeat(np.arange(n_subjects), 10)
    quarter = np.tile(np.arange(1, 11), n_subjects)
    enrolled = np.repeat(rng.binomial(1, 0.5, n_subjects), 10)
    post_launch = (quarter >= 6).astype(int)

    # Subject-level background variables (constant across quarters)
    age            = np.repeat(rng.integers(8, 18, n_subjects), 10)
    prior_placements = np.repeat(rng.poisson(2, n_subjects), 10)

    # Outcome: baseline trend + program effect after launch + noise
    baseline_trend = 0.15 * quarter
    program_effect = enrolled * post_launch * (-1.8)   # true ATT: -1.8 disruptions/quarter
    noise          = rng.normal(0, 0.8, len(quarter))
    placement_disruptions = np.clip(2 + baseline_trend + program_effect + 0.05 * prior_placements + noise, 0, None).round(1)

    return pd.DataFrame({
        "subject_id":           subject_ids,
        "quarter":              quarter,
        "program_enrolled":     enrolled,
        "post_launch":          post_launch,
        "age":                  age,
        "prior_placements":     prior_placements,
        "placement_disruptions": placement_disruptions,
    })

# ── Variable helpers ───────────────────────────────────────────────────────────
def suggest_variables(df):
    cols = df.columns.tolist()
    treatment_keywords = ["treat","program","enroll","intervention","mentoring","assigned","group","cohort","participant"]
    outcome_keywords   = ["outcome","result","disruption","success","score","rate","count","incident","event","measure"]
    # Exclude obvious non-variable columns from candidacy
    exclude_keywords   = ["id","time","date","year","quarter","month","week","period","post","before","after","wave"]
    candidates = [c for c in cols if not any(x in c.lower() for x in exclude_keywords)]
    t = next((c for c in candidates if any(x in c.lower() for x in treatment_keywords)), None)
    o = next((c for c in candidates if any(x in c.lower() for x in outcome_keywords) and c != t), None)
    # Fallbacks: avoid time/post columns
    if t is None: t = candidates[-2] if len(candidates) >= 2 else cols[-2]
    if o is None: o = candidates[-1] if candidates and candidates[-1] != t else cols[-1]
    return t, o, [c for c in cols if c not in (t, o)]

def ai_suggest_variables(question, columns):
    raw = _chat(f'Dataset columns: {columns}\nQuestion: "{question}"\n'
                'Return ONLY valid JSON (no fences) with keys: treatment, outcome, confounders.')
    if not raw: return None
    try:
        return json.loads(raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())
    except json.JSONDecodeError:
        return None

# ── Recommender ────────────────────────────────────────────────────────────────
def recommend_approach(df):
    cols = df.columns.str.lower()
    time_keywords = ["time", "date", "year", "quarter", "month", "week", "period", "wave", "visit", "survey"]
    post_keywords = ["post", "after", "launch", "intervention", "follow"]
    has_time = any(any(kw in c for kw in time_keywords) for c in cols)
    has_post = any(any(kw in c for kw in post_keywords) for c in cols)
    if has_time and has_post:
        return {"method": "Difference-in-Differences",
                "reason": "Your data tracks people over time and has a clear before/after intervention point."}
    if has_time:
        return {"method": "Interrupted Time Series",
                "reason": "Your data has a time dimension but no separate comparison group."}
    return {"method": "Backdoor (PSM / Regression)",
            "reason": "Your data is a single snapshot in time with background variables that may affect who received the program."}

# ── Data quality ───────────────────────────────────────────────────────────────
def check_readiness(df, treatment, outcome, confounders):
    issues = []
    if df[treatment].nunique() < 2: issues.append("The treatment column has only one value — there's no comparison group.")
    if df[outcome].nunique()   < 2: issues.append("The outcome column has only one value — there's nothing to measure change in.")
    missing = df[[treatment, outcome, *confounders]].isnull().sum().sum()
    if missing: issues.append(f"{int(missing)} missing values detected in key columns.")
    if len(df) < 100: issues.append(f"Small sample ({len(df)} rows) — estimates may be unreliable. At least 100 rows are recommended.")
    return issues

def auto_clean(df, treatment, outcome, confounders):
    df = df.dropna(subset=[treatment, outcome]).copy()
    for col in [treatment, outcome, *confounders]:
        try: df[col] = pd.to_numeric(df[col])
        except: pass
    rename = {c: c.strip().lower().replace(" ","_") for c in df.columns}
    return df.rename(columns=rename), rename

# ── Causal methods ─────────────────────────────────────────────────────────────
def build_graph(treatment, outcome, confounders):
    edges = [f"{treatment} -> {outcome};"]
    for c in confounders: edges += [f"{c} -> {treatment};", f"{c} -> {outcome};"]
    return "digraph {\n" + "\n".join(edges) + "\n}"

def run_did(df, treatment, outcome, time_col, post_col):
    for col in [treatment, outcome, time_col, post_col]:
        if col not in df.columns: raise ValueError(f"Missing column: {col}")
    if df[treatment].nunique() < 2: raise ValueError("Treatment needs 0 and 1.")
    if df[post_col].nunique()  < 2: raise ValueError("Post variable needs 0 and 1.")
    model = smf.ols(f"{outcome} ~ {treatment} + {post_col} + {treatment}:{post_col}", data=df).fit()
    ix = f"{treatment}:{post_col}"
    return model.params.get(ix), model.bse.get(ix), model

def check_parallel_trends(df, treatment, outcome, time_col, post_col):
    pre = df[df[post_col] == 0].copy().sort_values(time_col)
    tv  = time_col if pd.api.types.is_numeric_dtype(pre[time_col]) else "time_index"
    if tv == "time_index": pre["time_index"] = range(len(pre))
    m   = smf.ols(f"{outcome} ~ {tv} + {tv}:{treatment}", data=pre).fit()
    fig, ax = plt.subplots(figsize=(6, 3))
    for val, g in pre.groupby(treatment):
        ax.plot(g[time_col], g[outcome], label=f"{'Program' if val==1 else 'Comparison'} group", marker="o")
    ax.set(title="Pre-Program Trends (should look parallel)", xlabel=time_col, ylabel=outcome)
    ax.legend()
    return m.params.get(f"{tv}:{treatment}"), m.pvalues.get(f"{tv}:{treatment}"), fig

# ── Glossary tooltip helper ────────────────────────────────────────────────────
GLOSSARY = {
    "confounder":   "A background variable that influences both who gets the program AND the outcome — e.g. prior trauma affecting both program assignment and disruptions.",
    "PSM":          "Propensity Score Matching — pairs each program participant with a similar non-participant to estimate the program's effect.",
    "DiD":          "Difference-in-Differences — compares how the outcome changed over time in the program group vs a comparison group.",
    "ATE":          "Average Treatment Effect — the estimated average impact of the program across all participants.",
    "DAG":          "Directed Acyclic Graph — a diagram showing your assumptions about how variables relate causally.",
    "parallel trends": "The assumption in DiD that, without the program, both groups would have followed similar trends over time.",
}

def glossary_tip(term):
    return f"**{term}** — _{GLOSSARY.get(term, '')}_ "

# ══════════════════════════════════════════════════════════════════════════════
# UI — Title
# ══════════════════════════════════════════════════════════════════════════════
st.title("🧠 Causal Inference for Program Evaluation")
st.caption("Walk through the steps below to estimate whether your program caused a change in outcomes.")
if not _AI: st.info("💡 AI guidance is disabled — set the OPENAI_API_KEY environment variable to enable it.")

# ══════════════════════════════════════════════════════════════════════════════
# METHOD SELECTION QUIZ
# ══════════════════════════════════════════════════════════════════════════════

METHOD_INFO = {
    "Difference-in-Differences (DiD)": {
        "icon": "📊",
        "when": "You have data on the same people/units measured **before and after** a program launched, plus a comparison group that did not receive the program.",
        "strengths": "Controls for stable unobserved differences between groups. Widely accepted by funders and peer reviewers.",
        "watch_out": "Requires a clear program launch date, a valid comparison group, and that both groups were trending similarly before the program (parallel trends).",
        "sample": "Sample: Program Cohort (Panel/DiD)",
        "readable_name": "Difference-in-Differences",
    },
    "Propensity Score Matching / Regression (PSM)": {
        "icon": "🔵",
        "when": "You have a **single snapshot** of data (one observation per person), a program group, and a comparison group with background variables recorded.",
        "strengths": "Works with cross-sectional data. PSM creates a well-matched comparison group; regression controls statistically for background differences.",
        "watch_out": "Cannot account for unobserved selection factors. Needs a reasonably large sample (100+ per group) for reliable matching.",
        "sample": "Sample: Youth Mentoring (Cross-sectional)",
        "readable_name": "Propensity Score Matching / Regression",
    },
    "Interrupted Time Series (ITS)": {
        "icon": "📈",
        "when": "You have **aggregate data over time** (e.g. monthly case counts for a whole agency) with a clear program start date, but no separate comparison group.",
        "strengths": "Doesn't require individual-level data or a comparison group. Good for policy changes affecting an entire population.",
        "watch_out": "Can't rule out other events that happened at the same time as the program. Needs at least 8–10 time points before and after.",
        "sample": "Upload CSV",
        "readable_name": "Interrupted Time Series",
    },
    "Regression Discontinuity (RDD)": {
        "icon": "📐",
        "when": "Program eligibility is determined by a **score or threshold** (e.g. risk score above 7, age under 12, income below $X). People just above and below the cutoff are compared.",
        "strengths": "Very credible — people near the cutoff are similar by design. Widely respected in policy evaluation.",
        "watch_out": "Only estimates the effect near the cutoff, not across the whole population. Requires a large enough sample near the threshold.",
        "sample": "Upload CSV",
        "readable_name": "Regression Discontinuity",
    },
}

def run_quiz():
    st.markdown("Answer a few questions about your data and program. We'll recommend the best method and explain why.")
    st.divider()

    # Q1
    q1 = st.radio(
        "**1. How is your data structured?**",
        ["Each row is one person, measured once (a single snapshot)",
         "Each row is one person at one point in time — the same people appear multiple times",
         "Each row is an aggregate measure (e.g. monthly totals for a whole agency or region)"],
        index=None, key="q1"
    )

    if not q1:
        st.info("👆 Answer each question to get your recommendation.")
        return

    # Q2
    q2 = st.radio(
        "**2. Do you have a comparison group** — people or units that did NOT receive the program?",
        ["Yes — I have both program participants and a comparison group",
         "No — everyone in my data received the program"],
        index=None, key="q2"
    )
    if not q2: return

    # Q3 — only relevant for panel/aggregate data
    q3 = None
    if "multiple times" in q1 or "aggregate" in q1:
        q3 = st.radio(
            "**3. Do you have data from BEFORE the program started?**",
            ["Yes — I have observations both before and after the program launched",
             "No — my data only covers the period after the program started"],
            index=None, key="q3"
        )
        if not q3: return

    # Q4 — eligibility cutoff
    q4 = st.radio(
        "**4. Was program eligibility determined by a score or threshold?**  \n_(e.g. a risk score above a certain number, age cutoff, income limit)_",
        ["Yes — there is a clear numeric cutoff that determined who qualified",
         "No — eligibility was based on other factors or was discretionary"],
        index=None, key="q4"
    )
    if not q4: return

    # Q5 — sample size
    q5 = st.radio(
        "**5. Roughly how many people are in your dataset?**",
        ["Fewer than 50", "50 – 200", "200 – 1,000", "More than 1,000"],
        index=None, key="q5"
    )
    if not q5: return

    st.divider()

    # ── Decision logic ──────────────────────────────────────────────────────────
    warnings = []
    n_small = q5 == "Fewer than 50"
    n_medium = q5 == "50 – 200"

    if n_small:
        warnings.append("⚠️ **Very small sample (< 50):** All causal methods will have wide uncertainty. Results should be treated as preliminary only.")
    if n_medium:
        warnings.append("⚠️ **Small sample (50–200):** PSM matching may struggle to find good matches. Regression adjustment is more reliable at this size.")

    if "Yes — there is a clear numeric cutoff" in q4:
        method = "Regression Discontinuity (RDD)"
        warnings.append("💡 RDD is not yet implemented in this tool — but it's the strongest choice for your situation. Consider it for a future analysis.")

    elif "aggregate" in q1:
        method = "Interrupted Time Series (ITS)"
        if "No — my data only covers" in (q3 or ""):
            warnings.append("⚠️ Without pre-program data, ITS cannot estimate a trend change. Consider whether earlier records are available.")
        warnings.append("💡 ITS is not yet implemented in this tool. It's on the roadmap — use PSM/Regression as a starting point if you have individual-level data.")

    elif "multiple times" in q1 and "Yes — I have both" in q2:
        if q3 and "Yes — I have observations both" in q3:
            method = "Difference-in-Differences (DiD)"
        else:
            method = "Interrupted Time Series (ITS)"
            warnings.append("⚠️ Without pre-program observations, DiD is not possible. ITS is the next best option if you have a comparison group trend.")

    elif "snapshot" in q1 and "Yes — I have both" in q2:
        method = "Propensity Score Matching / Regression (PSM)"

    elif "No — everyone" in q2:
        method = "Interrupted Time Series (ITS)"
        warnings.append("⚠️ Without a comparison group, it's difficult to isolate the program's effect. ITS requires pre/post time-series data.")
        if "snapshot" in q1:
            warnings.append("🔴 With a single snapshot and no comparison group, causal inference is very limited. Consider whether historical data or a waitlist group is available.")

    else:
        method = "Propensity Score Matching / Regression (PSM)"

    # ── Display result ──────────────────────────────────────────────────────────
    info = METHOD_INFO[method]
    st.subheader(f"{info['icon']} Recommended Method: {method}")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown(f"**When to use it:**  \n{info['when']}")
        st.markdown(f"**Strengths:**  \n{info['strengths']}")
    with col_r:
        st.markdown(f"**Watch out for:**  \n{info['watch_out']}")

        # Implementability flag
        implemented = method in ["Difference-in-Differences (DiD)", "Propensity Score Matching / Regression (PSM)"]
        if implemented:
            st.success("✅ This method is available in this tool. Continue to Step 1 to load your data.")
        else:
            st.warning("🔧 This method is not yet implemented — it's on the roadmap. PSM/Regression is available as a starting point.")

    for w in warnings:
        st.warning(w)

    # Save result to session state
    st.session_state.quiz_result = method
    st.session_state.quiz_done = True

    # Comparison table
    with st.expander("📋 Compare all methods side by side", expanded=False):
        rows = []
        for name, m in METHOD_INFO.items():
            rows.append({
                "Method": f"{m['icon']} {m['readable_name']}",
                "Best for": m["when"].replace("**",""),
                "In this tool": "✅ Yes" if name in ["Difference-in-Differences (DiD)", "Propensity Score Matching / Regression (PSM)"] else "🔧 Roadmap",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── Quiz UI ────────────────────────────────────────────────────────────────────
with st.expander("🧭 Not sure which method to use? Take the 2-minute quiz", 
                 expanded=not st.session_state.quiz_done):
    run_quiz()

if st.session_state.quiz_done and st.session_state.quiz_result:
    info = METHOD_INFO.get(st.session_state.quiz_result, {})
    st.info(f"📌 Your quiz result: **{st.session_state.quiz_result}** — continue below to load your data.")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load Data
# ══════════════════════════════════════════════════════════════════════════════
st.header("Step 1 — Load Your Data")

src = st.radio("Choose data source:", ["Upload CSV", "Sample: Youth Mentoring (Cross-sectional)", "Sample: Program Cohort (Panel/DiD)"],
               horizontal=True)

if src == "Sample: Youth Mentoring (Cross-sectional)":
    df_raw = generate_backdoor_data()
    st.success("Sample dataset loaded — 1,000 youth records with mentoring program and placement disruption outcome.")
elif src == "Sample: Program Cohort (Panel/DiD)":
    df_raw = generate_did_data()
    st.success("Sample dataset loaded — panel data with treatment group, time, and post-intervention indicator.")
else:
    f = st.file_uploader("Upload your CSV file", type=["csv"])
    if not f:
        st.info("👆 Upload a CSV to get started. Each row should represent one person or one observation period.")
        st.stop()
    df_raw = pd.read_csv(f)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Understand Your Data
# ══════════════════════════════════════════════════════════════════════════════
st.header("Step 2 — Understand Your Data")

df = st.session_state.df_clean if st.session_state.df_clean is not None else df_raw

col_prev, col_stats = st.columns([3, 1])
with col_prev:
    st.subheader("Preview")
    st.dataframe(df.head(10), use_container_width=True)
with col_stats:
    st.subheader("At a glance")
    st.metric("Rows", f"{len(df):,}")
    st.metric("Columns", len(df.columns))
    missing = df.isnull().sum().sum()
    st.metric("Missing values", int(missing), delta=None if missing == 0 else "⚠️ present", delta_color="inverse")

if _AI:
    with st.expander("🧠 What does this data represent? (AI explanation)", expanded=False):
        # Auto-run explanation when expander is opened, cache in session state
        if st.session_state.dataset_explanation is None:
            with st.spinner("Analysing your dataset…"):
                st.session_state.dataset_explanation = _chat(
                    f'Explain this dataset plainly to a program director (not a statistician). '
                    f'Columns: {df.columns.tolist()}, rows: {len(df)}, '
                    f'dtypes: {df.dtypes.astype(str).to_dict()}, '
                    f'sample values: {df.head(3).to_dict()}. '
                    f'Cover: (1) what this dataset likely represents, '
                    f'(2) which column is probably the program/treatment, '
                    f'(3) which column is probably the outcome, '
                    f'(4) any obvious data quality concerns. Use plain language, no jargon.'
                )
        if st.session_state.dataset_explanation:
            st.write(st.session_state.dataset_explanation)
        if st.button("🔄 Re-explain"):
            st.session_state.dataset_explanation = None
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Clean Your Data
# ══════════════════════════════════════════════════════════════════════════════
st.header("Step 3 — Clean Your Data")

if st.session_state.df_clean is not None:
    c1, c2 = st.columns([4, 1])
    c1.success("✅ Using cleaned dataset")
    if c2.button("↩ Undo clean"): st.session_state.df_clean = None; st.rerun()
else:
    st.write("Auto-clean will: remove rows with missing program/outcome data, convert text numbers to numeric, and standardise column names.")
    if st.button("✨ Clean My Data", type="primary"):
        # Need treatment/outcome to be set for cleaning — use heuristic defaults if not yet set
        t_default, o_default, c_default = suggest_variables(df_raw)
        t_clean = st.session_state.treatment or t_default
        o_clean = st.session_state.outcome   or o_default
        c_clean = st.session_state.confounders or c_default
        df_c, rmap = auto_clean(df_raw, t_clean, o_clean, c_clean)
        st.session_state.update(df_clean=df_c,
                                dataset_explanation=None,   # reset so AI re-explains cleaned data
                                treatment=rmap.get(t_clean, t_clean),
                                outcome=rmap.get(o_clean, o_clean),
                                confounders=[rmap.get(c, c) for c in c_clean])
        st.success(f"✅ Cleaned — {len(df_raw)-len(df_c)} rows removed, {len(df_c):,} rows remaining.")
        with st.expander("🔍 Column name changes"): st.json(rmap)
        st.rerun()

# Refresh df after potential clean
df = st.session_state.df_clean if st.session_state.df_clean is not None else df_raw

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Define Your Model
# ══════════════════════════════════════════════════════════════════════════════
st.header("Step 4 — Define Your Model")

# ── 4a: AI question ────────────────────────────────────────────────────────────
if _AI:
    st.subheader("4a — Describe your evaluation question (optional)")
    st.caption("Type your question in plain language and AI will suggest which columns to use.")
    with st.form("qform"):
        q = st.text_input("e.g. Did the mentoring program reduce placement disruptions?",
                          value=st.session_state.question)
        if st.form_submit_button("✨ Suggest Variables") and q:
            st.session_state.question = q
            with st.spinner("Interpreting your question…"):
                res = ai_suggest_variables(q, df.columns.tolist())
            valid = df.columns.tolist()
            if res and res.get("treatment") in valid and res.get("outcome") in valid:
                st.session_state.update(treatment=res["treatment"], outcome=res["outcome"],
                                        confounders=[c for c in res.get("confounders", []) if c in valid])
                st.success("Variables suggested — review and adjust below.")
                exp = _chat(f'A program director asked: "{q}". '
                            f'We set up: treatment={res["treatment"]}, outcome={res["outcome"]}, '
                            f'confounders={res.get("confounders",[])}. '
                            f'Explain in 3 short bullet points why these choices make sense causally. '
                            f'Use plain language, no equations.')
                if exp: st.info(exp)
                st.json(res)
            else:
                st.warning("Couldn't match your question to columns — please select variables manually below.")
                t, o, c = suggest_variables(df)
                st.session_state.update(treatment=t, outcome=o, confounders=c)

# ── 4b: Manual variable selection ─────────────────────────────────────────────
st.subheader("4b — Select variables")

# Glossary inline
with st.expander("📖 What do these terms mean?", expanded=False):
    for term in ["confounder", "PSM", "DiD", "DAG"]:
        st.markdown(glossary_tip(term))

cols = df.columns.tolist()
if st.session_state.treatment not in cols: st.session_state.treatment = cols[0]
if st.session_state.outcome   not in cols: st.session_state.outcome   = cols[1] if len(cols)>1 else cols[0]
st.session_state.confounders = [c for c in st.session_state.confounders if c in cols]

col_t, col_o, col_c = st.columns(3)
with col_t:
    treatment = st.selectbox("🟢 Program / Treatment variable",
                             cols, index=cols.index(st.session_state.treatment),
                             help="The column that indicates whether someone received the program (usually 0/1).")
with col_o:
    outcome = st.selectbox("🔴 Outcome variable",
                           cols, index=cols.index(st.session_state.outcome),
                           help="The result you're trying to change — e.g. placement disruptions, test scores.")
with col_c:
    confounders = st.multiselect("🔵 Background variables (confounders)",
                                 [c for c in cols if c not in (treatment, outcome)],
                                 default=[c for c in st.session_state.confounders if c not in (treatment, outcome)],
                                 help="Variables that affect both who got the program AND the outcome — e.g. age, prior history.")

st.session_state.update(treatment=treatment, outcome=outcome, confounders=confounders)

if treatment == outcome:
    st.error("❌ Program and outcome must be different columns."); st.stop()
if treatment in confounders or outcome in confounders:
    st.warning("⚠️ Your program or outcome variable is also listed as a background variable — please remove it from that list.")

# ── 4c: Live DAG ───────────────────────────────────────────────────────────────
st.subheader("4c — Your causal diagram")
st.caption("This diagram shows your assumptions about how variables relate. Arrows mean 'influences'.")
dot = Digraph()
dot.attr("node", shape="box", style="filled", fontname="Helvetica")
dot.attr(rankdir="LR")
dot.node(treatment, f"Program\n({treatment})", color="lightgreen")
dot.node(outcome,   f"Outcome\n({outcome})",   color="lightcoral")
dot.edge(treatment, outcome, label=" effect?")
for c in confounders:
    dot.node(c, c, color="lightblue")
    dot.edge(c, treatment, style="dashed")
    dot.edge(c, outcome,   style="dashed")
st.graphviz_chart(dot)
st.caption("🟢 Green = program   🔴 Red = outcome   🔵 Blue = background variable   Dashed = confounding path")

# ── 4d: Data readiness ─────────────────────────────────────────────────────────
st.subheader("4d — Data readiness check")
issues = check_readiness(df, treatment, outcome, confounders)
if not issues:
    st.success("✅ Data looks ready for analysis")
else:
    for i in issues: st.warning(f"⚠️ {i}")
    if _AI and st.button("🛠 How do I fix these issues?"):
        with st.spinner():
            fixes = _chat(f"A program director has these data issues: {issues}. "
                          f"Suggest simple fixes in plain language. Assume they use Excel or basic Python.")
        if fixes: st.write(fixes)
    st.info("You can still proceed, but results may be less reliable.")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Choose Analysis Method
# ══════════════════════════════════════════════════════════════════════════════
st.header("Step 5 — Analysis Method")

rec = recommend_approach(df)
st.info(f"**Recommended method:** {rec['method']}  \n**Why:** {rec['reason']}")

with st.expander("📖 What does this method do?", expanded=False):
    descriptions = {
        "Difference-in-Differences": (
            "**Difference-in-Differences (DiD)** compares how your outcome changed over time "
            "in the program group vs a comparison group. It's like asking: 'Did things improve "
            "more for program participants than for similar people who didn't participate?' "
            f"\n\n{glossary_tip('parallel trends')}"
        ),
        "Backdoor (PSM / Regression)": (
            "**Propensity Score Matching (PSM)** finds people who didn't receive the program "
            "but were otherwise very similar to those who did, then compares outcomes. "
            "**Linear Regression** estimates the program effect while statistically controlling "
            "for background variables. Running both gives you a sense of how robust the result is."
        ),
        "Interrupted Time Series": (
            "**Interrupted Time Series** looks at a single group's trend before and after the "
            "program started, and tests whether the trend changed meaningfully at that point."
        ),
    }
    st.markdown(descriptions.get(rec["method"], ""))

# ── DiD extra setup ────────────────────────────────────────────────────────────
did_ready = time_col = post_col = None
if rec["method"] == "Difference-in-Differences":
    st.subheader("DiD column setup")
    st.caption("Select which columns represent time and the before/after split.")

    with st.expander("❓ How do I know which column is which?", expanded=False):
        st.markdown("""
**For DiD you need four distinct columns:**

| Column | What it means | Typical values | Example |
|--------|--------------|----------------|---------|
| 🟢 **Program variable** | Was this person/group in the program? | 0 or 1 | `group = 1` (enrolled) |
| 🔴 **Outcome variable** | What you're measuring | Any number | `disruptions = 3` |
| 🕐 **Time variable** | When was this observation recorded? | Month, quarter, year | `time = 4` (4th month) |
| 📅 **Post indicator** | Was this observation after the program started? | 0 or 1 | `post = 1` (after launch) |

**Common mistake:** Using the outcome as the post indicator, or using time as the outcome.
The post indicator is just a flag (0/1) — it does not measure results.
        """)
        if _AI and st.button("🧠 Help me identify these columns in my data", key="did_help"):
            with st.spinner("Analysing your columns…"):
                hint = _chat(
                    f"A program director is setting up a Difference-in-Differences analysis. "
                    f"Their dataset has these columns: {df.columns.tolist()}. "
                    f"Sample values: {df.head(3).to_dict()}. "
                    f"Identify which column is likely: (1) the program/treatment indicator, "
                    f"(2) the outcome, (3) the time variable, (4) the post-intervention indicator. "
                    f"If any are ambiguous, say so. Use plain language, one short paragraph per column."
                )
            if hint: st.info(hint)

    # Guard: treatment and outcome must be confirmed before we can filter correctly
    if not treatment or not outcome or treatment == outcome:
        st.warning("⚠️ Please confirm your program and outcome variables in Step 4 before setting up DiD columns.")
    else:
        did_cols = [c for c in cols if c not in (treatment, outcome)]
        if len(did_cols) < 2:
            st.error("❌ Not enough columns — time and post indicator must be separate from your program and outcome columns.")
            st.stop()

        # Clear any stale session state that points to an excluded column
        if st.session_state.get("did_time") not in did_cols:
            st.session_state["did_time"] = did_cols[0]

        c1, c2 = st.columns(2)
        with c1:
            time_col = st.selectbox(
                "Time variable", did_cols,
                index=did_cols.index(st.session_state["did_time"]),
                key="did_time",
                help="The column representing when each observation occurred (e.g. month, quarter, year). Cannot be your program or outcome column.")

        post_cols = [c for c in did_cols if c != time_col]
        if st.session_state.get("did_post") not in post_cols:
            st.session_state["did_post"] = post_cols[0]

        with c2:
            post_col = st.selectbox(
                "Post-intervention indicator (0 = before, 1 = after)",
                post_cols,
                index=post_cols.index(st.session_state["did_post"]),
                key="did_post",
                help="A 0/1 column: 0 = before the program started, 1 = after. Must differ from your program, outcome, and time columns.")

        if df[post_col].nunique() < 2:
            st.warning("The post column must contain both 0 (before) and 1 (after) values.")
        else:
            did_ready = True
            st.success("✅ DiD setup looks valid")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Run Analysis
# ══════════════════════════════════════════════════════════════════════════════
st.header("Step 6 — Run Analysis")
st.caption("When you're happy with the setup above, click the button to estimate the program's effect.")

if not st.button("▶ Estimate Program Effect", type="primary"): st.stop()

try:
    # Treatment group sizes
    tcounts = df[treatment].value_counts()
    min_grp = int(tcounts.min())
    st.session_state.min_group = min_grp

    if tcounts.nunique() < 2: st.error("❌ Treatment column needs at least two groups (0 and 1)."); st.stop()
    if df[outcome].nunique() < 2: st.error("❌ Outcome column needs at least two different values."); st.stop()
    if min_grp < 10: st.warning("⚠️ One group has fewer than 10 people — estimates will be very uncertain.")

    use_did = rec["method"] == "Difference-in-Differences" and did_ready and time_col and post_col
    st.session_state.use_did = use_did

    progress = st.progress(0, text="Starting analysis…")

    # ── DiD ────────────────────────────────────────────────────────────────────
    if use_did:
        progress.progress(30, text="Running Difference-in-Differences model…")
        eff, se, _ = run_did(df, treatment, outcome, time_col, post_col)
        st.session_state.update(did_effect=eff, did_se=se)

        progress.progress(70, text="Checking parallel trends…")
        _, pval, fig = check_parallel_trends(df, treatment, outcome, time_col, post_col)
        progress.progress(100, text="Done ✅"); progress.empty()

        st.subheader("📈 Pre-program trend check")
        st.caption("For DiD to be valid, both groups should have been moving similarly before the program started.")
        st.pyplot(fig, use_container_width=False)
        if pval is not None:
            if pval > 0.05:
                st.success(f"🟢 Trends look parallel before the program — DiD assumption holds (p={pval:.3f}).")
            else:
                st.warning(f"🟡 Trends may not be parallel before the program (p={pval:.3f}). Interpret results cautiously.")
        else:
            st.info("Not enough pre-program data to check trends.")

        diag_df = pd.DataFrame({"Method": ["Difference-in-Differences"],
                                 "Estimated Effect": [eff], "Std Error": [se], "N": [len(df)]})
        st.subheader("📊 Results")
        st.dataframe(diag_df, use_container_width=True)

    # ── Backdoor ───────────────────────────────────────────────────────────────
    else:
        progress.progress(20, text="Building causal model…")
        graph   = build_graph(treatment, outcome, confounders)
        cm      = CausalModel(data=df, treatment=treatment, outcome=outcome, graph=graph)
        estimand= cm.identify_effect()

        progress.progress(50, text="Running propensity score matching (this can take a moment)…")
        psm_r   = cm.estimate_effect(estimand, method_name="backdoor.propensity_score_matching")

        progress.progress(75, text="Running regression…")
        reg_r   = cm.estimate_effect(estimand, method_name="backdoor.linear_regression")

        progress.progress(90, text="Running refutation test…")
        psm_eff, reg_eff = psm_r.value, reg_r.value
        st.session_state.update(psm_effect=psm_eff, reg_effect=reg_eff)

        st.subheader("📊 Results — Two Methods for Robustness")
        diag_df = pd.DataFrame({"Method": ["Propensity Score Matching (PSM)", "Linear Regression"],
                                 "Estimated Effect": [psm_eff, reg_eff]})
        st.dataframe(diag_df, use_container_width=True)

        diff = abs(psm_eff - reg_eff)
        if   diff < 0.05: st.success("🟢 Both methods agree — result is more reliable.")
        elif diff < 0.15: st.warning("🟡 Methods differ slightly — result is suggestive but check your assumptions.")
        else:             st.info("🔍 Methods differ substantially — this may mean the program effect varies across participants, or background variables need review.")

        st.subheader("🔍 Refutation test")
        st.caption("This test randomly scrambles the program assignment. A trustworthy model should show near-zero effect after scrambling.")
        try:
            refut = cm.refute_estimate(estimand, psm_r, method_name="placebo_treatment_refuter")
            st.write(refut)
        except Exception:
            st.info("Refutation test could not run — this sometimes happens with small samples.")

        if _AI:
            with st.expander("🧠 AI: Plain-language explanation of these results", expanded=False):
                with st.spinner():
                    exp = _chat(
                        f'A program director ran a causal analysis. '
                        f'Question: "{st.session_state.question or "program impact on outcome"}". '
                        f'PSM estimated effect: {psm_eff:.4f}. Regression: {reg_eff:.4f}. '
                        f'Outcome: {outcome}. Treatment: {treatment}. '
                        f'Explain in plain language: what do these numbers mean, do the methods agree, '
                        f'and what should the director tell stakeholders? No jargon, no equations.'
                    )
                if exp: st.write(exp)

        progress.progress(100, text="Done ✅"); progress.empty()

    # ── Practical interpretation ───────────────────────────────────────────────
    st.subheader("🎯 What does this mean in practice?")
    baseline = df[outcome].mean()
    is_binary_outcome = df[outcome].nunique() == 2

    col_base, col_effect = st.columns(2)
    with col_base:
        if is_binary_outcome:
            baseline_pct = baseline * 100
            st.metric("Baseline rate (no program)", f"{baseline_pct:.1f}%",
                      help=f"The share of people with {outcome}=1 among those who did not receive the program.")
        else:
            st.metric(f"Average {outcome} (no program)", f"{baseline:.2f}",
                      help="The average outcome value before accounting for the program effect.")

    if use_did:
        e = st.session_state.did_effect or 0
        direction = "lower" if e < 0 else "higher"
        pct_change = (e / baseline * 100) if baseline != 0 else 0
        per_100 = abs(e * 100) if is_binary_outcome else None

        with col_effect:
            st.metric(f"Program effect on {outcome}", f"{e:+.3f}",
                      delta=f"{pct_change:+.1f}% vs baseline",
                      delta_color="inverse" if e < 0 else "normal")

        if is_binary_outcome:
            st.success(
                f"**The program {'reduced' if e < 0 else 'increased'} {outcome} by "
                f"{abs(pct_change):.1f} percentage points** compared to the comparison group "
                f"over the same period. That's roughly **{abs(e*100):.0f} fewer cases per 100 people** enrolled."
            )
        else:
            st.success(
                f"**The program {'reduced' if e < 0 else 'increased'} {outcome} by "
                f"{abs(e):.3f} units** on average ({abs(pct_change):.1f}% change from baseline), "
                f"compared to the comparison group over the same time period."
            )

        with st.expander("💬 How to explain this to a funder or board", expanded=False):
            n_program = int((df[treatment] == 1).sum())
            total_impact = abs(e) * n_program
            st.markdown(f"""
> *"Our analysis compared {outcome} trends in the program group and a similar comparison group
> before and after the program launched. The program group showed a
> {'reduction' if e < 0 else 'an increase'} of **{abs(e):.3f} units** in {outcome}
> that was not seen in the comparison group. Across our {n_program} program participants,
> this represents an estimated total change of **{total_impact:.1f} units** in {outcome}."*

*Always add: "This is an observational estimate, not from a randomised trial."*
            """)

    else:
        p, r = st.session_state.psm_effect or 0, st.session_state.reg_effect or 0
        avg  = (p + r) / 2
        direction = "lower" if avg < 0 else "higher"
        pct_change = (avg / baseline * 100) if baseline != 0 else 0

        with col_effect:
            st.metric(f"Estimated program effect", f"{avg:+.3f}",
                      delta=f"{pct_change:+.1f}% vs baseline",
                      delta_color="inverse" if avg < 0 else "normal")

        if is_binary_outcome:
            st.success(
                f"**The program {'reduced' if avg < 0 else 'increased'} {outcome} by approximately "
                f"{abs(pct_change):.1f} percentage points** after accounting for background differences. "
                f"That translates to roughly **{abs(avg*100):.0f} fewer cases per 100 similar people** enrolled."
            )
        else:
            st.success(
                f"**The program {'reduced' if avg < 0 else 'increased'} {outcome} by approximately "
                f"{abs(avg):.3f} units** ({abs(pct_change):.1f}% change from the baseline average of {baseline:.2f}), "
                f"after accounting for background differences between groups."
            )

        st.caption(f"PSM estimate: {p:.4f}  |  Regression estimate: {r:.4f}  |  Average used above: {avg:.4f}")

        with st.expander("💬 How to explain this to a funder or board", expanded=False):
            n_program = int((df[treatment] == 1).sum())
            total_impact = abs(avg) * n_program
            st.markdown(f"""
> *"We compared {n_program} program participants to similar non-participants using two
> statistical methods (propensity score matching and regression). Both suggest the program
> {'reduced' if avg < 0 else 'increased'} {outcome} by approximately **{abs(avg):.3f} units**
> ({abs(pct_change):.1f}% change). Across all participants, this represents an estimated
> total impact of **{total_impact:.1f} units** in {outcome}."*

*Always add: "This is an observational estimate, not from a randomised trial."*
            """)

    # ── Confidence summary ─────────────────────────────────────────────────────
    st.subheader("🧭 How confident should you be?")
    signals = []
    mn = st.session_state.min_group
    if mn: signals.append("good_balance" if mn > 100 else "moderate_balance" if mn > 30 else "poor_balance")
    if not use_did:
        d = abs((st.session_state.psm_effect or 0) - (st.session_state.reg_effect or 0))
        signals.append("strong_agreement" if d < 0.05 else "moderate_agreement" if d < 0.15 else "weak_agreement")

    if   "poor_balance" in signals or "weak_agreement"         in signals:
        st.error("🔴 Lower confidence — small groups or inconsistent methods. Share results carefully and note limitations.")
    elif "moderate_balance" in signals or "moderate_agreement" in signals:
        st.warning("🟡 Moderate confidence — results are suggestive. Useful for internal decision-making but not definitive proof.")
    else:
        st.success("🟢 Higher confidence — results are consistent across methods and groups are well-balanced.")

    st.caption("⚠️ These results come from observational data, not a randomised trial. The estimate assumes no hidden confounders.")

    with st.expander("📊 Technical diagnostics", expanded=False):
        st.dataframe(diag_df, use_container_width=True)
        st.write(f"**Treatment group sizes:**")
        st.dataframe(tcounts.rename_axis(treatment).reset_index(name="count"), use_container_width=True)
        st.caption("PSM matches on propensity scores from the observed background variables. Regression controls for them linearly.")

    # ── Export ─────────────────────────────────────────────────────────────────
    st.subheader("📄 Export Report")
    results_txt = (f"DiD Effect: {st.session_state.did_effect:.4f}  SE: {st.session_state.did_se:.4f}"
                   if use_did else
                   f"PSM: {st.session_state.psm_effect:.4f}  |  Regression: {st.session_state.reg_effect:.4f}")
    report = (f"CAUSAL ANALYSIS REPORT\n{'='*40}\n"
              f"Question:    {st.session_state.question or 'N/A'}\n"
              f"Treatment:   {treatment}\nOutcome:     {outcome}\n"
              f"Confounders: {', '.join(confounders) or 'None'}\n\n"
              f"TREATMENT GROUP SIZES\n{tcounts.to_string()}\n\n"
              f"RESULTS\n{results_txt}\n\n"
              f"LIMITATIONS\n"
              f"- Observational data, not a randomised trial\n"
              f"- Assumes no important unmeasured confounders\n"
              f"- Results depend on model assumptions and variable selection\n"
              f"- Use as one input to decision-making, not definitive proof\n")
    st.download_button("⬇️ Download Report", report, "causal_analysis_report.txt", "text/plain")
    st.session_state.analysis_ran = True

except Exception as exc:
    st.error("Something went wrong during the analysis. See details below.")
    with st.expander("🔧 Technical error details (for your data team)"):
        st.exception(exc)