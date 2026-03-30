import json, os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import streamlit as st
from graphviz import Digraph

# dowhy has C-extension dependencies (cvxpy, econml) that may fail on
# newer Python versions. Degrade gracefully rather than crashing on startup.
try:
    from dowhy import CausalModel
    _DOWHY = True
except Exception as _dowhy_err:
    _DOWHY = False
    CausalModel = None

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
    """
    Realistic cross-sectional PSM dataset: youth mentoring program.
    Design goals:
    - ~50/50 treatment split for good PSM overlap
    - Propensity scores spanning 0.15-0.85 (real common support)
    - Confounders with moderate SMDs (0.15-0.35) — correctable by PSM
    - True ATT = -0.9 disruption units (mentoring reduces disruptions)
    - Binary outcome (disruption: 0/1) for interpretability
    """
    rng = np.random.default_rng(seed)

    # Background variables (confounders)
    age              = rng.integers(8, 18, n).astype(float)
    trauma_score     = rng.normal(50, 15, n).clip(0, 100)
    prior_placements = rng.poisson(2, n).clip(0, 8).astype(float)
    case_complexity  = rng.binomial(1, 0.4, n).astype(float)  # 1=complex

    # Standardise for logit — keeps coefficients interpretable
    trauma_z = (trauma_score - 50) / 15
    prior_z  = (prior_placements - 2) / 1.5
    age_z    = (age - 13) / 3

    # Program assignment: higher trauma/prior placements → more likely referred
    # Intercept=0 gives ~50% baseline enrollment; range ~0.2-0.8
    logit    = 0.35*trauma_z + 0.30*prior_z + 0.15*case_complexity - 0.10*age_z
    mentoring= rng.binomial(1, 1/(1+np.exp(-logit)))

    # Outcome: disruption (binary). True causal effect = -0.9 units (mentoring helps)
    noise           = rng.normal(0, 0.8, n)
    disruption_cont = 1.5 + 0.6*trauma_z + 0.5*prior_z + 0.3*case_complexity - 0.9*mentoring + noise
    disruption      = (disruption_cont > disruption_cont.mean()).astype(int)

    return pd.DataFrame({
        "age":              age.astype(int),
        "trauma_score":     trauma_score.round(1),
        "prior_placements": prior_placements.astype(int),
        "case_complexity":  case_complexity.astype(int),
        "mentoring":        mentoring,
        "disruption":       disruption,
    })

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
    outcome_keywords   = ["outcome","result","disruption","success","rate","count","incident","event","measure","index"]
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
# ── Causal methods ─────────────────────────────────────────────────────────────
def build_graph(treatment, outcome, confounders):
    edges = [f"{treatment} -> {outcome};"]
    for c in confounders: edges += [f"{c} -> {treatment};", f"{c} -> {outcome};"]
    return "digraph {\n" + "\n".join(edges) + "\n}"


def _detect_subject_col(df, treatment, outcome, time_col, post_col):
    """Heuristically find a subject/unit ID column for clustering."""
    exclude = {treatment, outcome, time_col, post_col}
    id_kws  = ["id", "subject", "person", "client", "case", "unit", "individual", "youth"]
    for c in df.columns:
        if c in exclude: continue
        if any(k in c.lower() for k in id_kws): return c
    return None


def run_did(df, treatment, outcome, time_col, post_col):
    """
    Two-way fixed effects DiD with clustered SEs by subject.
    Falls back to HC3 robust SEs if no subject ID column is found.
    Returns: (effect, se, ci_low, ci_high, pvalue, model)
    """
    for col in [treatment, outcome, time_col, post_col]:
        if col not in df.columns: raise ValueError(f"Missing column: {col}")
    if df[treatment].nunique() < 2: raise ValueError("Treatment needs 0 and 1.")
    if df[post_col].nunique()  < 2: raise ValueError("Post variable needs 0 and 1.")

    subject_col = _detect_subject_col(df, treatment, outcome, time_col, post_col)
    ix = f"{treatment}:{post_col}"

    if subject_col and df[subject_col].nunique() > 1:
        formula = f"{outcome} ~ {treatment} + {post_col} + {ix} + C({subject_col}) + C({time_col})"
        try:
            model = smf.ols(formula, data=df).fit(
                cov_type="cluster", cov_kwds={"groups": df[subject_col]}
            )
        except Exception:
            formula = f"{outcome} ~ {treatment} + {post_col} + {ix}"
            model   = smf.ols(formula, data=df).fit(cov_type="HC3")
    else:
        formula = f"{outcome} ~ {treatment} + {post_col} + {ix}"
        model   = smf.ols(formula, data=df).fit(cov_type="HC3")

    eff  = model.params.get(ix)
    se   = model.bse.get(ix)
    ci   = model.conf_int().loc[ix] if ix in model.conf_int().index else (None, None)
    pval = model.pvalues.get(ix)
    return eff, se, ci[0], ci[1], pval, model


def check_parallel_trends(df, treatment, outcome, time_col, post_col):
    """
    Pre-trend test on group-period means (avoids repeated-measures inflation).
    Returns (slope_diff_coef, pvalue, fig, n_pre_periods)
    """
    pre = df[df[post_col] == 0].copy()
    tv  = time_col if pd.api.types.is_numeric_dtype(pre[time_col]) else "time_index"
    if tv == "time_index":
        pre["time_index"] = pre[time_col].rank(method="dense").astype(int)

    grp_means = pre.groupby([tv, treatment])[outcome].mean().reset_index()
    n_pre = grp_means[tv].nunique()

    fig, ax = plt.subplots(figsize=(6, 3))
    colors = {1: "#2ecc71", 0: "#e74c3c"}
    for val, g in grp_means.groupby(treatment):
        label = "Program" if val == 1 else "Comparison"
        ax.plot(g[tv], g[outcome], marker="o", color=colors[val],
                label=f"{label} group", linewidth=2)

    if n_pre < 3:
        ax.set(title="Pre-Program Trends (fewer than 3 periods — test unreliable)",
               xlabel=time_col, ylabel=outcome)
        ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
        return None, None, fig, n_pre

    m    = smf.ols(f"{outcome} ~ {tv} + {tv}:{treatment}", data=grp_means).fit()
    coef = m.params.get(f"{tv}:{treatment}")
    pval = m.pvalues.get(f"{tv}:{treatment}")

    # Add fitted trend lines
    for val, g in grp_means.groupby(treatment):
        t_range   = np.linspace(g[tv].min(), g[tv].max(), 50)
        intercept = m.params.get("Intercept", 0)
        slope     = m.params.get(tv, 0)
        slope_int = m.params.get(f"{tv}:{treatment}", 0) if val == 1 else 0
        ax.plot(t_range, intercept + (slope + slope_int) * t_range,
                "--", color=colors[val], alpha=0.5)

    ax.set(title="Pre-Program Trends (should look parallel)", xlabel=time_col, ylabel=outcome)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3); fig.tight_layout()
    return coef, pval, fig, n_pre


def compute_psm_balance(df, treatment, confounders):
    """
    Standardised Mean Differences (SMD) for each confounder before matching.
    SMD < 0.1 = good balance; 0.1-0.2 = moderate; > 0.2 = poor.
    """
    rows = []
    treated = df[df[treatment] == 1]
    control = df[df[treatment] == 0]
    for c in confounders:
        try:
            mt, mc   = treated[c].mean(), control[c].mean()
            st_, sc  = treated[c].std(),  control[c].std()
            pooled   = np.sqrt((st_**2 + sc**2) / 2)
            smd      = (mt - mc) / pooled if pooled > 0 else 0
            balance  = "✅ Good" if abs(smd) < 0.1 else "🟡 Moderate" if abs(smd) < 0.2 else "🔴 Poor"
            rows.append({"Variable": c, "Mean (Program)": round(mt, 3),
                         "Mean (Comparison)": round(mc, 3),
                         "SMD": round(abs(smd), 3), "Balance": balance})
        except Exception:
            pass
    return pd.DataFrame(rows)


def plot_propensity_overlap(df, treatment, confounders):
    """
    Fit logistic propensity model and plot score histograms by group.
    Overlap between distributions indicates common support.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X = df[confounders].select_dtypes(include=[np.number]).dropna()
    y = df.loc[X.index, treatment]
    if X.empty or y.nunique() < 2: return None

    try:
        scaler = StandardScaler()
        lr     = LogisticRegression(max_iter=1000)
        lr.fit(scaler.fit_transform(X), y)
        scores = lr.predict_proba(scaler.transform(X))[:, 1]
    except Exception:
        return None

    fig, ax = plt.subplots(figsize=(6, 3))
    for val, label, color in [(1, "Program group", "#2ecc71"),
                               (0, "Comparison group", "#e74c3c")]:
        ax.hist(scores[y == val], bins=25, alpha=0.5,
                color=color, label=label, density=True)
    ax.set(title="Propensity Score Overlap (Common Support)",
           xlabel="Propensity Score (probability of program enrollment)",
           ylabel="Density")
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    return fig


def compute_evalue(effect, se):
    """
    E-value (VanderWeele & Ding 2017): minimum unmeasured confounding strength
    needed to fully explain away the observed effect.
    Uses the continuous-outcome approximation: convert standardised effect (d = effect/se)
    to an approximate risk ratio, then apply the E-value formula.
    RR_approx = exp(0.477 * |d|)  — matches VanderWeele 2017 Table 2 for d in [0,2].
    E-value = RR + sqrt(RR*(RR-1)).  Range: 1 (no robustness) to ~5 (very robust).
    Returns (evalue_point, evalue_ci) or (None, None) if not computable.
    """
    if se is None or se == 0 or effect is None: return None, None
    d = abs(effect / se)          # standardised effect size (like Cohen's d)
    if d < 0.001: return None, None   # indistinguishable from zero

    # Approximate RR on the confounder-association scale
    # Capped at d=3 to avoid astronomical values from poorly estimated SEs
    d_capped = min(d, 3.0)
    rr = np.exp(0.477 * d_capped)
    if rr <= 1: return None, None

    evalue = rr + np.sqrt(rr * (rr - 1))

    # CI bound E-value: use effect - 1.96*se
    d_ci = max(abs(effect) - 1.96 * se, 0) / se if se > 0 else 0
    if d_ci < 0.001:
        evalue_ci = 1.0   # CI includes null — result not significant, E-value for CI = 1
    else:
        d_ci_capped = min(d_ci, 3.0)
        rr_ci = np.exp(0.477 * d_ci_capped)
        evalue_ci = rr_ci + np.sqrt(max(rr_ci * (rr_ci - 1), 0)) if rr_ci > 1 else 1.0

    return round(evalue, 2), round(evalue_ci, 2)

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
    # ── Pre-flight checks ──────────────────────────────────────────────────────
    tcounts = df[treatment].value_counts()
    min_grp = int(tcounts.min())
    max_grp = int(tcounts.max())
    n_total = int(tcounts.sum())
    imbalance_ratio = min_grp / max_grp if max_grp > 0 else 0
    st.session_state.min_group = min_grp

    if df[treatment].nunique() < 2:
        st.error("❌ Treatment column needs at least two groups (0 and 1)."); st.stop()
    if df[outcome].nunique() < 2:
        st.error("❌ Outcome column needs at least two different values."); st.stop()
    if min_grp < 10:
        st.warning("⚠️ One group has fewer than 10 people — estimates will be very uncertain.")
    if imbalance_ratio < 0.1:
        st.warning(f"⚠️ Severe group imbalance: {min_grp} vs {max_grp} — PSM matching will struggle. "
                   f"Consider whether a comparison group is truly available.")

    use_did = rec["method"] == "Difference-in-Differences" and did_ready and time_col and post_col
    st.session_state.use_did = use_did
    progress = st.progress(0, text="Starting analysis…")

    # ══════════════════════════════════════════════════════════════════════════
    # DiD PATH
    # ══════════════════════════════════════════════════════════════════════════
    if use_did:
        progress.progress(30, text="Running two-way fixed effects DiD model…")
        eff, se, ci_lo, ci_hi, pval, did_model = run_did(
            df, treatment, outcome, time_col, post_col)
        st.session_state.update(did_effect=eff, did_se=se)

        progress.progress(70, text="Checking parallel trends on group means…")
        trend_coef, trend_pval, trend_fig, n_pre = check_parallel_trends(
            df, treatment, outcome, time_col, post_col)
        progress.progress(100, text="Done ✅"); progress.empty()

        # ── Parallel trends ────────────────────────────────────────────────────
        st.subheader("📈 Assumption Check: Parallel Trends")
        st.caption(
            "DiD is only valid if both groups were moving similarly **before** the program. "
            "The chart shows group-period means (averaging within each group and time period). "
            "Dashed lines show the fitted pre-trend for each group — they should look parallel."
        )
        col_plot, col_verdict = st.columns([2, 1])
        with col_plot:
            st.pyplot(trend_fig, use_container_width=False)
        with col_verdict:
            if n_pre < 3:
                st.warning(f"⚠️ Only {n_pre} pre-program period(s). Need at least 3 to test trends reliably. Treat this assumption as unverified.")
            elif trend_pval is not None:
                if trend_pval > 0.1:
                    st.success(f"🟢 **Parallel trends plausible** — p = {trend_pval:.3f}. The slope difference between groups is not statistically significant before the program.")
                elif trend_pval > 0.05:
                    st.warning(f"🟡 **Borderline** — p = {trend_pval:.3f}. Some evidence of diverging trends. Interpret DiD results with caution.")
                else:
                    st.error(f"🔴 **Parallel trends likely violated** — p = {trend_pval:.3f}. Groups were trending differently before the program. DiD estimates may be biased.")
            st.caption(f"Pre-program periods available: **{n_pre}** (≥ 3 recommended for reliable testing)")

        # ── Results ────────────────────────────────────────────────────────────
        st.subheader("📊 Results — Difference-in-Differences")
        st.caption("Standard errors are clustered by subject (where an ID column was detected) or HC3 robust. "
                   "This corrects for repeated observations on the same person.")

        c1, c2, c3, c4 = st.columns(4)
        sig_label = "p < 0.05 ✅" if pval < 0.05 else "p ≥ 0.05 — not significant"
        c1.metric("Estimated Effect (ATT)", f"{eff:+.3f}")
        c2.metric("95% CI", f"[{ci_lo:.3f}, {ci_hi:.3f}]")
        c3.metric("Std Error", f"{se:.3f}")
        c4.metric("p-value", f"{pval:.3f}", delta=sig_label,
                  delta_color="normal" if pval < 0.05 else "inverse")

        diag_df = pd.DataFrame({
            "Method": ["DiD (Two-Way FE)"], "Effect (ATT)": [round(eff, 4)],
            "SE": [round(se, 4)], "CI low": [round(ci_lo, 4)],
            "CI high": [round(ci_hi, 4)], "p-value": [round(pval, 4)], "N": [n_total]})

        # ── E-value ────────────────────────────────────────────────────────────
        ev, ev_ci = compute_evalue(eff, se)
        st.subheader("🛡️ Sensitivity: E-value")
        st.caption(
            "The E-value answers: *How strong would an unmeasured confounder need to be "
            "to fully explain away this result?* Higher = more robust. "
            "An E-value of 2.0 means a confounder would need to double the odds of both "
            "program enrollment AND the outcome to explain the finding away."
        )
        if ev:
            col_ev1, col_ev2 = st.columns(2)
            col_ev1.metric("E-value (point estimate)", str(ev),
                           help="Unmeasured confounding strength needed to explain away the effect.")
            col_ev2.metric("E-value (confidence limit)", str(ev_ci),
                           help="Unmeasured confounding needed to shift the CI to include zero.")
            if ev >= 2.0:
                st.success(f"🟢 E-value = {ev} — the result is relatively robust. "
                           f"An unmeasured confounder would need to be quite strong to explain it away.")
            elif ev >= 1.5:
                st.warning(f"🟡 E-value = {ev} — moderate robustness. "
                           f"A moderately strong unmeasured confounder could explain this result.")
            else:
                st.error(f"🔴 E-value = {ev} — low robustness. "
                         f"Even a weak unmeasured confounder could explain this result.")
        else:
            st.info("E-value could not be computed (effect may not be distinguishable from zero).")

    # ══════════════════════════════════════════════════════════════════════════
    # BACKDOOR PATH (PSM + Regression)
    # ══════════════════════════════════════════════════════════════════════════
    else:
        progress.progress(10, text="Building causal model…")
        if not _DOWHY:
            st.error("❌ The causal modelling library (dowhy) could not be loaded. "
                     "This is a Python 3.14 compatibility issue. Pin Python 3.11 in deployment settings.")
            st.stop()

        # ── Assumption: propensity overlap ─────────────────────────────────────
        st.subheader("📐 Assumption Check: Propensity Score Overlap")
        st.caption(
            "Before matching, we check whether the program and comparison groups have "
            "overlapping propensity scores — i.e. for every program participant, "
            "there exist similar non-participants. If distributions don't overlap, "
            "the model is extrapolating rather than comparing like with like."
        )
        if confounders:
            overlap_fig = plot_propensity_overlap(df, treatment, confounders)
            if overlap_fig:
                col_ov1, col_ov2 = st.columns([2, 1])
                with col_ov1:
                    st.pyplot(overlap_fig, use_container_width=False)
                with col_ov2:
                    st.markdown("""
**How to read this chart:**
- Both distributions should overlap substantially in the middle
- 🟢 Good overlap = matching is valid
- 🔴 Little overlap = estimates may not be reliable

If the program group scores are all near 1.0 and the comparison group near 0.0, there's essentially no common support.
                    """)
            else:
                st.info("Overlap plot could not be generated (no numeric confounders found).")
        else:
            st.warning("No confounders selected — propensity score overlap cannot be assessed.")

        # ── Assumption: covariate balance ──────────────────────────────────────
        st.subheader("⚖️ Assumption Check: Covariate Balance")
        st.caption(
            "Standardised Mean Differences (SMD) show how similar the groups are on each "
            "background variable **before** matching. SMD < 0.1 is the standard threshold for "
            "good balance. Large imbalances may bias the estimate even after PSM."
        )
        if confounders:
            bal_df = compute_psm_balance(df, treatment, confounders)
            if not bal_df.empty:
                st.dataframe(bal_df, use_container_width=True, hide_index=True)
                poor_balance = bal_df[bal_df["SMD"] > 0.2]
                if poor_balance.empty:
                    st.success("🟢 All confounders show reasonable pre-matching balance (SMD < 0.2).")
                else:
                    st.warning(f"🟡 {len(poor_balance)} variable(s) show poor balance (SMD > 0.2): "
                               f"{', '.join(poor_balance['Variable'].tolist())}. "
                               f"PSM will attempt to correct this, but large imbalances are harder to fix.")
        else:
            st.warning("No confounders selected — balance cannot be assessed.")

        # ── Run models ─────────────────────────────────────────────────────────
        graph    = build_graph(treatment, outcome, confounders)
        cm       = CausalModel(data=df, treatment=treatment, outcome=outcome, graph=graph)
        estimand = cm.identify_effect()

        progress.progress(45, text="Running propensity score matching (ATT)…")
        psm_r  = cm.estimate_effect(estimand, method_name="backdoor.propensity_score_matching")

        progress.progress(70, text="Running OLS regression (ATE)…")
        reg_r  = cm.estimate_effect(estimand, method_name="backdoor.linear_regression")

        progress.progress(85, text="Running refutation test…")
        psm_eff, reg_eff = psm_r.value, reg_r.value
        st.session_state.update(psm_effect=psm_eff, reg_effect=reg_eff)

        # ── Results ────────────────────────────────────────────────────────────
        st.subheader("📊 Results — PSM and Regression")
        st.caption(
            "These two methods answer related but subtly different questions depending on implementation. "
            "**PSM here uses nearest-neighbour matching anchored to the treated group**, which estimates the ATT "
            "(Average Treatment effect on the Treated — the effect for people who actually enrolled). "
            "PSM can also estimate the ATE by matching in both directions, but that is not the default used here. "
            "**OLS regression with controls** estimates a variance-weighted average of individual effects — "
            "close to the ATE under linearity and homogeneity, but not identical. "
            "Both are reported separately — do not average them."
        )

        col_psm, col_reg = st.columns(2)
        with col_psm:
            st.markdown("### Propensity Score Matching")
            st.markdown("**Estimates: ATT** (as implemented here — nearest-neighbour, treated-anchored)")
            st.metric("Estimated Effect", f"{psm_eff:+.4f}")
            st.caption("Each enrolled participant is matched to a similar non-participant. "
                       "The effect is estimated for the enrolled group. PSM *can* estimate ATE "
                       "by matching in both directions, but this app uses the ATT formulation.")
        with col_reg:
            st.markdown("### Linear Regression with Controls")
            st.markdown("**Estimates: weighted average ≈ ATE** (under linearity and homogeneity)")
            st.metric("Estimated Effect", f"{reg_eff:+.4f}")
            st.caption("Controls for confounders linearly across the full sample. Approximates the ATE "
                       "but puts more weight on units near the centre of the covariate distribution. "
                       "Strictly an ATE only if the treatment effect is constant across all participants.")

        # Agreement note
        diff = abs(psm_eff - reg_eff)
        st.markdown("---")
        st.markdown("**Do the two estimates agree?**")
        if diff < 0.1:
            st.success(f"🟢 The two estimates are close ({diff:.4f} apart). This is reassuring — it suggests "
                       f"the program effect is fairly consistent across participants, and that the ATT "
                       f"(effect on enrolled youth) is similar to the broader population-average effect.")
        else:
            st.info(f"🔵 The estimates differ by {diff:.4f}. This can arise for several reasons: "
                    f"(1) the program effect genuinely varies — people who enrolled may have benefited "
                    f"{'more' if psm_eff < reg_eff else 'less'} than the average eligible person would have; "
                    f"(2) the linearity assumption in regression may not hold; or "
                    f"(3) PSM matching quality may be imperfect. Neither estimate is necessarily wrong — "
                    f"they answer slightly different questions.")

        diag_df = pd.DataFrame({
            "Method": ["PSM (nearest-neighbour)", "OLS Regression with controls"],
            "Estimand": ["ATT — avg effect on enrolled", "Weighted avg ≈ ATE"],
            "Estimated Effect": [round(psm_eff, 4), round(reg_eff, 4)]})

        # ── Refutation ─────────────────────────────────────────────────────────
        st.subheader("🔍 Refutation Test (Placebo Treatment)")
        st.caption(
            "The program assignment is randomly scrambled. A robust model should show a "
            "near-zero effect after scrambling — if it doesn't, the original estimate may be "
            "picking up chance patterns rather than a real program effect."
        )
        try:
            refut = cm.refute_estimate(estimand, psm_r, method_name="placebo_treatment_refuter")
            st.write(refut)
        except Exception:
            st.info("Refutation test could not run — this sometimes happens with small samples.")

        # ── E-value ────────────────────────────────────────────────────────────
        st.subheader("🛡️ Sensitivity: E-value (PSM estimate)")
        st.caption(
            "The E-value answers: *How strong would an unmeasured confounder need to be "
            "to fully explain away this result?* Higher = more robust to hidden confounding."
        )
        # Use PSM SE approximation from effect / sqrt(n)
        # SE approximation for PSM: use outcome SD / sqrt(n_treated) as conservative estimate
        # This is the standard error of a difference in means, which PSM approximates
        n_treated   = int((df[treatment] == 1).sum())
        outcome_sd  = df[outcome].std()
        psm_se_approx = outcome_sd / np.sqrt(max(n_treated, 1)) * np.sqrt(2)  # two-sample
        ev, ev_ci = compute_evalue(psm_eff, psm_se_approx)
        if ev:
            col_ev1, col_ev2 = st.columns(2)
            col_ev1.metric("E-value (point estimate)", str(ev))
            col_ev2.metric("E-value (CI bound)", str(ev_ci))
            if ev >= 2.0:
                st.success(f"🟢 E-value = {ev}. The result is relatively robust to unmeasured confounding.")
            elif ev >= 1.5:
                st.warning(f"🟡 E-value = {ev}. A moderately strong unmeasured confounder could explain this result.")
            else:
                st.error(f"🔴 E-value = {ev}. Even weak unmeasured confounders could explain this result.")
        else:
            st.info("E-value could not be computed.")

        if _AI:
            with st.expander("🧠 AI: Plain-language explanation of these results", expanded=False):
                with st.spinner():
                    exp = _chat(
                        f'''A program director ran a causal analysis.
Question: "{st.session_state.question or "program impact on outcome"}".
PSM effect (nearest-neighbour, ATT): {psm_eff:.4f}. Regression effect (weighted avg approx ATE): {reg_eff:.4f}.
Outcome: {outcome}. Treatment: {treatment}.
Explain: (1) what ATT vs ATE means in plain language for this program,
(2) what the numbers mean, (3) whether they tell a consistent story,
(4) what the director should tell stakeholders. No jargon, no equations.'''
                    )
                if exp: st.write(exp)

        progress.progress(100, text="Done ✅"); progress.empty()

    # ══════════════════════════════════════════════════════════════════════════
    # PRACTICAL INTERPRETATION (both paths)
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("🎯 What does this mean in practice?")
    baseline        = df[outcome].mean()
    is_binary       = df[outcome].nunique() == 2
    effect_for_interp = eff if use_did else psm_eff   # ATT is more relevant for program directors

    col_base, col_eff = st.columns(2)
    with col_base:
        label = "Baseline rate" if is_binary else f"Average {outcome}"
        val   = f"{baseline*100:.1f}%" if is_binary else f"{baseline:.2f}"
        st.metric(label, val, help="Observed average in the full sample before accounting for program effects.")
    with col_eff:
        # For binary outcomes, effect IS the percentage point change directly.
        # For continuous, express as % of baseline mean.
        if is_binary:
            pp_change = effect_for_interp * 100   # e.g. -0.268 → -26.8 pp
            delta_label = f"{pp_change:+.1f} percentage points"
        else:
            pct_chg   = (effect_for_interp / baseline * 100) if baseline != 0 else 0
            delta_label = f"{pct_chg:+.1f}% change from baseline"

        st.metric("Estimated program effect",
                  f"{effect_for_interp:+.3f}",
                  delta=delta_label,
                  delta_color="inverse" if effect_for_interp < 0 else "normal")

    direction = "reduced" if effect_for_interp < 0 else "increased"
    if is_binary:
        st.success(
            f"**The program {direction} {outcome} by {abs(pp_change):.1f} percentage points** "
            f"— roughly {abs(pp_change):.0f} fewer cases per 100 {'enrolled participants' if use_did else 'similar people'}."
        )
    else:
        st.success(
            f"**The program {direction} {outcome} by {abs(effect_for_interp):.3f} units** "
            f"({abs(pct_chg):.1f}% change from the baseline of {baseline:.2f}). "
            f"{'This is the ATT — the effect for people who actually enrolled.' if not use_did else ''}"
        )

    with st.expander("💬 How to explain this to a funder or board", expanded=False):
        n_enrolled = int((df[treatment] == 1).sum())
        # Count unique subjects if possible to avoid double-counting panel rows
        subject_col = _detect_subject_col(df, treatment, outcome,
                                          time_col or "", post_col or "")
        if subject_col:
            n_enrolled = int(df[df[treatment]==1][subject_col].nunique())
        st.markdown(f"""
> *"We used observational causal analysis to estimate the impact of {treatment} on {outcome}.
> {'The Difference-in-Differences method compared trends before and after the program in the enrolled group vs a comparison group.' if use_did else 'Propensity score matching compared enrolled participants to similar non-participants.'}
> The analysis suggests the program **{direction} {outcome} by {'%.1f percentage points' % abs(pp_change) if is_binary else '%.3f units' % abs(effect_for_interp)}**
> {'(%.1f pp change from a baseline rate of %.1f%%)' % (abs(pp_change), baseline*100) if is_binary else '(%.1f%% change from baseline)' % abs(pct_chg if not is_binary else 0)},
> for the {n_enrolled} enrolled participants."*

**Important caveats to always include:**
- This is an observational estimate, not a randomised trial
- Results assume no important unmeasured differences between groups
- The estimate applies to people similar to those in this dataset
        """)
        st.caption("⚠️ Do not report a 'total population impact' by multiplying effect × N — "
                   "this assumes a constant, additive effect across all participants and is rarely justified.")

    # ── Confidence summary ─────────────────────────────────────────────────────
    st.subheader("🧭 How confident should you be?")
    signals = []

    # Group balance (ratio-based, not just minimum)
    if imbalance_ratio < 0.1:    signals.append("poor_balance")
    elif imbalance_ratio < 0.33: signals.append("moderate_balance")
    else:                         signals.append("good_balance")

    # Statistical significance
    if use_did:
        signals.append("significant" if pval < 0.05 else "not_significant")
        if n_pre < 3: signals.append("weak_trends_test")
    else:
        # Check confounder balance quality
        if confounders:
            bal_df2 = compute_psm_balance(df, treatment, confounders)
            if not bal_df2.empty:
                max_smd = bal_df2["SMD"].max()
                signals.append("good_balance_psm" if max_smd < 0.1
                                else "moderate_balance_psm" if max_smd < 0.2
                                else "poor_balance_psm")

    low_conf    = "poor_balance" in signals or "not_significant" in signals or "poor_balance_psm" in signals
    med_conf    = "moderate_balance" in signals or "weak_trends_test" in signals or "moderate_balance_psm" in signals

    if low_conf:
        st.error("🔴 **Lower confidence** — group imbalance, non-significant result, or failed assumption checks. "
                 "Share results carefully and prominently note limitations.")
    elif med_conf:
        st.warning("🟡 **Moderate confidence** — results are suggestive. Useful for internal learning "
                   "but not as definitive evidence of impact.")
    else:
        st.success("🟢 **Higher confidence** — groups are balanced, the result is statistically significant, "
                   "and assumption checks look reasonable.")

    st.caption("⚠️ All results come from observational data. Even high confidence here does not equal "
               "a randomised trial — unmeasured confounders may still be present.")

    # ── Technical diagnostics ──────────────────────────────────────────────────
    with st.expander("📊 Full technical diagnostics", expanded=False):
        st.dataframe(diag_df, use_container_width=True, hide_index=True)
        st.markdown("**Treatment group sizes:**")
        st.dataframe(tcounts.rename_axis(treatment).reset_index(name="count"),
                     use_container_width=True, hide_index=True)
        st.markdown(f"**Group balance ratio:** {imbalance_ratio:.2f} "
                    f"(1.0 = perfectly balanced; < 0.1 = severely imbalanced)")
        if use_did:
            st.caption(f"DiD model used {'two-way fixed effects with clustered SEs' if _detect_subject_col(df, treatment, outcome, time_col, post_col) else 'HC3 robust SEs'}.")
        else:
            st.caption("PSM uses nearest-neighbour matching anchored to treated units — estimates ATT. Regression uses the full sample with linear control for confounders — approximates ATE under homogeneity. PSM can also estimate ATE by bidirectional matching, but that is not implemented here.")

    # ── Export ─────────────────────────────────────────────────────────────────
    st.subheader("📄 Export Report")
    if use_did:
        results_txt = (f"Method: Difference-in-Differences (Two-Way FE)\n"
                       f"Effect (ATT): {eff:.4f}\n"
                       f"95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]\n"
                       f"Std Error: {se:.4f}  |  p-value: {pval:.4f}\n"
                       f"Pre-periods: {n_pre}  |  Parallel trends p: {trend_pval if trend_pval else 'N/A'}\n"
                       f"E-value: {ev if ev else 'N/A'}")
    else:
        results_txt = (f"PSM (nearest-neighbour, ATT): {psm_eff:.4f}\n"
                       f"Regression with controls (weighted avg ≈ ATE): {reg_eff:.4f}\n"
                       f"Difference between estimates: {diff:.4f}\n"
                       f"E-value (PSM): {ev if ev else 'N/A'}")

    report = (f"CAUSAL ANALYSIS REPORT\n{'='*40}\n"
              f"Question:    {st.session_state.question or 'N/A'}\n"
              f"Treatment:   {treatment}\nOutcome:     {outcome}\n"
              f"Confounders: {', '.join(confounders) or 'None'}\n\n"
              f"TREATMENT GROUP SIZES\n{tcounts.to_string()}\n"
              f"Group balance ratio: {imbalance_ratio:.2f}\n\n"
              f"RESULTS\n{results_txt}\n\n"
              f"ASSUMPTION CHECKS\n"
              f"- Parallel trends: {'Tested — see app for chart' if use_did else 'N/A for PSM'}\n"
              f"- Propensity overlap: {'N/A' if use_did else 'See app for chart'}\n"
              f"- Covariate balance (max SMD): {'N/A' if use_did else str(round(bal_df2['SMD'].max(), 3)) if confounders and not bal_df2.empty else 'No confounders'}\n\n"
              f"LIMITATIONS\n"
              f"- Observational data, not a randomised trial\n"
              f"- Assumes no important unmeasured confounders\n"
              f"- Results depend on model assumptions and variable selection\n"
              f"- PSM (nearest-neighbour) estimates ATT; regression approximates ATE under linearity\n""- These answer related but subtly different questions — do not average them\n"
              f"- Use as one input to decision-making, not definitive proof\n")
    st.download_button("⬇️ Download Report", report, "causal_analysis_report.txt", "text/plain")
    st.session_state.analysis_ran = True

except Exception as exc:
    st.error("Something went wrong during the analysis. See details below.")
    with st.expander("🔧 Technical error details (for your data team)"):
        st.exception(exc)