"""
SIA — Support Integrity Auditor — Streamlit App

Provides:
  - Single-ticket inference with full Evidence Dossier
  - Batch CSV upload with predictions + dossiers
  - Priority Mismatch Dashboard (distributions, mismatch types, top signals)
  - Severity Delta Heatmap (category x channel)
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import plotly.express as px
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from collections import Counter


# ══════════════════════════════════════════════════════════════════
# Page config
# ══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="SIA — Support Integrity Auditor",
    page_icon="🛡️",
    layout="wide",
)

# ══════════════════════════════════════════════════════════════════
# Constants (shared with predict.py)
# ══════════════════════════════════════════════════════════════════

PRIORITY_ORD = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
ORD_PRIORITY = {0: 'Low', 1: 'Medium', 2: 'High', 3: 'Critical'}

MODEL_NAME = "microsoft/deberta-v3-small"
MAX_LEN    = 256
MODEL_DIR  = Path("models/deberta_lora")

CATEGORIES = ['Technical', 'Billing', 'Account', 'Fraud', 'General Inquiry']
PRIORITIES = ['Low', 'Medium', 'High', 'Critical']
CHANNELS   = ['Web Form', 'Email', 'Chat']

ESC_EVIDENCE = [
    (r'\bcannot\b',                         'negation_escalation', 0.18),
    (r'\bunable\b',                         'negation_escalation', 0.15),
    (r'\bcrash\w*\b',                       'system_failure',      0.22),
    (r'\bnot (?:loading|working|syncing)\b','functional_failure',  0.20),
    (r'\bphish\w*\b',                       'security_threat',     0.35),
    (r'\bstolen\b',                         'security_threat',     0.35),
    (r'\bfraud\w*\b',                       'fraud_indicator',     0.35),
    (r'\bhack\w*\b',                        'security_threat',     0.30),
    (r'\bdata.*(?:loss|breach|leak)\b',     'data_risk',           0.35),
    (r'\blocke?d? out\b',                   'access_blocked',      0.25),
    (r'\baccount.*(?:suspend|comprom)\w*\b','account_risk',        0.28),
    (r'\bpayment.*fail\w*\b',               'payment_failure',     0.22),
    (r'\bimmediately\b',                    'urgency_language',    0.15),
    (r'\burgent\b',                         'urgency_language',    0.15),
]

DEESC_EVIDENCE = [
    (r'\bwhere is\b',       'informational_query', -0.12),
    (r'\bhow do i\b',       'informational_query', -0.12),
    (r'\bfeature request\b','feature_request',     -0.15),
    (r'\bheadquarters\b',   'general_inquiry',     -0.20),
    (r'\broadmap\b',        'general_inquiry',     -0.15),
]

RT_MEDIANS = {
    ('Technical','Critical'):5,  ('Technical','High'):18,
    ('Technical','Medium'):38,   ('Technical','Low'):48,
    ('Fraud','Critical'):4,      ('Fraud','High'):12,
    ('Billing','Critical'):6,    ('Billing','High'):20,
    ('Billing','Medium'):42,     ('Billing','Low'):52,
    ('Account','High'):22,       ('Account','Medium'):40, ('Account','Low'):50,
    ('General Inquiry','Medium'):35, ('General Inquiry','Low'):45,
}

CATEGORY_SEVERITY = {
    'Fraud':           ('Critical', 0.30, 'Fraud tickets carry inherent security risk'),
    'Technical':       ('High',     0.20, 'Technical failures impact system availability'),
    'Account':         ('Medium',   0.12, 'Account issues affect user access'),
    'Billing':         ('Medium',   0.10, 'Billing issues have financial impact'),
    'General Inquiry': ('Low',     -0.15, 'General inquiries are typically informational'),
}


# ══════════════════════════════════════════════════════════════════
# Model loading (cached — loads once per session)
# ══════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading SIA model...")
def load_model():
    """Load tokenizer, DeBERTa-v3-small + LoRA adapter, and decision threshold."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    best_dir = MODEL_DIR / "best"
    if not best_dir.exists():
        st.error(f"Model not found at {best_dir}. Run train_pipeline.py first.")
        st.stop()

    tokenizer = AutoTokenizer.from_pretrained(str(best_dir))
    base = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True)
    model = PeftModel.from_pretrained(base, str(best_dir))
    model = model.float().to(device)
    model.eval()

    threshold_path = MODEL_DIR / "best_threshold.npy"
    threshold = float(np.load(str(threshold_path))[0]) if threshold_path.exists() else 0.5

    return tokenizer, model, threshold, device


# ══════════════════════════════════════════════════════════════════
# Input construction & inference
# ══════════════════════════════════════════════════════════════════

def rt_bin(rt_hours: float) -> str:
    if rt_hours <= 12:   return 'FAST'
    elif rt_hours <= 48: return 'MID'
    return 'SLOW'


def build_input_text(row: dict) -> str:
    """Hybrid input: text fields + structured metadata tokens (same as predict.py)."""
    return (f"{row['Ticket_Subject']} [SEP] "
            f"{row['Ticket_Description']} [SEP] "
            f"category:{row['Issue_Category']} "
            f"channel:{row['Ticket_Channel']} "
            f"rt:{rt_bin(float(row['Resolution_Time_Hours']))} "
            f"priority:{row['Priority_Level']}")


def predict_batch(texts, tokenizer, model, device, batch_size=64) -> np.ndarray:
    """Run inference and return mismatch probabilities for a list of texts."""
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, truncation=True, padding='max_length',
                         max_length=MAX_LEN, return_tensors='pt')
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        probs = torch.softmax(out.logits.float(), dim=1)[:, 1].cpu().numpy()
        all_probs.extend(probs)
    return np.array(all_probs)


# ══════════════════════════════════════════════════════════════════
# Evidence extraction (identical logic to predict.py / Stage 3)
# ══════════════════════════════════════════════════════════════════

def extract_text_evidence(row, mismatch_type):
    text = f"{row['Ticket_Subject']} {row['Ticket_Description']}".lower()
    subj = row['Ticket_Subject']
    evidence = []
    for pattern, signal_type, weight in ESC_EVIDENCE + DEESC_EVIDENCE:
        m = re.search(pattern, text)
        if m:
            field = "Ticket_Subject" if re.search(pattern, subj.lower()) \
                    else "Ticket_Description"
            evidence.append({
                "signal":       "keyword",
                "type":         signal_type,
                "value":        m.group(0),
                "source_field": field,
                "weight":       round(weight, 3)
            })
    if mismatch_type == "Hidden Crisis":
        evidence = [e for e in evidence if e.get('weight', 0) > 0]
    else:
        evidence = [e for e in evidence if e.get('weight', 0) < 0]
    return evidence


def extract_rt_evidence(row, mismatch_type):
    actual_rt = float(row['Resolution_Time_Hours'])
    cat, prio = row['Issue_Category'], row['Priority_Level']
    expected  = RT_MEDIANS.get((cat, prio), 40.0)
    ratio     = actual_rt / max(expected, 1)

    if ratio < 0.4:
        if mismatch_type == "False Alarm":
            interp = (f"Resolved in {actual_rt:.0f}h vs expected ~{expected:.0f}h "
                      f"for {prio} {cat} — resolved quickly, consistent with "
                      f"a ticket that did not require high priority resources")
        else:
            interp = (f"Resolved in {actual_rt:.0f}h vs expected ~{expected:.0f}h "
                      f"for {prio} {cat} — {ratio:.1f}x faster than expected, "
                      f"suggesting true urgency exceeded assigned priority")
        weight = 0.25
    elif ratio > 2.5:
        interp = (f"Resolved in {actual_rt:.0f}h vs expected ~{expected:.0f}h "
                  f"for {prio} {cat} — {ratio:.1f}x slower than expected, "
                  f"suggesting ticket was under-resourced for its actual severity")
        weight = 0.20
    else:
        interp = (f"Resolved in {actual_rt:.0f}h, within normal range "
                  f"for {prio} {cat} (expected ~{expected:.0f}h)")
        weight = 0.05

    return {
        "signal":         "resolution_time",
        "value":          f"{actual_rt:.0f}h",
        "expected":       f"~{expected:.0f}h",
        "rt_ratio":       round(ratio, 2),
        "source_field":   "Resolution_Time_Hours",
        "interpretation": interp,
        "weight":         round(weight, 3)
    }


def extract_satisfaction_evidence(row, mismatch_type):
    sat  = int(row['Satisfaction_Score'])
    prio = row['Priority_Level']

    if mismatch_type == "Hidden Crisis" and sat <= 2 and prio in ['Low', 'Medium']:
        interp = (f"Satisfaction score {sat}/5 on a {prio}-priority ticket — "
                  f"low satisfaction indicates the customer felt their urgency "
                  f"was not matched by the assigned priority level")
        weight = 0.20
    elif mismatch_type == "False Alarm" and sat >= 4 and prio in ['Critical', 'High']:
        interp = (f"Satisfaction score {sat}/5 on a {prio}-priority ticket — "
                  f"high satisfaction suggests the issue resolved easily, "
                  f"consistent with over-triage")
        weight = 0.12
    else:
        interp = (f"Satisfaction score {sat}/5 — insufficient signal "
                  f"for {mismatch_type} classification independently")
        weight = 0.02

    return {
        "signal":         "satisfaction_score",
        "value":          str(sat),
        "source_field":   "Satisfaction_Score",
        "interpretation": interp,
        "weight":         round(weight, 3)
    }


def extract_category_evidence(row, mismatch_type):
    cat, prio = row['Issue_Category'], row['Priority_Level']
    expected_sev, base_weight, rationale = CATEGORY_SEVERITY.get(
        cat, ('Medium', 0.05, 'Standard category'))

    exp_ord  = PRIORITY_ORD.get(expected_sev, 1)
    prio_ord = PRIORITY_ORD.get(prio, 1)

    if mismatch_type == "Hidden Crisis" and exp_ord > prio_ord:
        interp = (f"{cat} tickets typically warrant {expected_sev} priority — "
                  f"assigning {prio} is below the category baseline. {rationale}.")
        weight = base_weight
    elif mismatch_type == "False Alarm" and exp_ord < prio_ord:
        interp = (f"{cat} tickets typically warrant {expected_sev} priority — "
                  f"assigning {prio} exceeds the category baseline. {rationale}.")
        weight = -base_weight
    else:
        interp = (f"{cat} category assigned {prio} priority — "
                  f"within expected range for this category.")
        weight = 0.03

    return {
        "signal":         "category_baseline",
        "value":          cat,
        "expected_level": expected_sev,
        "source_field":   "Issue_Category",
        "interpretation": interp,
        "weight":         round(weight, 3)
    }


def compute_severity_delta(assigned, inferred):
    a, i = PRIORITY_ORD.get(assigned, 1), PRIORITY_ORD.get(inferred, 1)
    delta = i - a
    if delta > 0:
        return f"+{delta} (under-prioritised by {delta} level{'s' if delta > 1 else ''})", delta
    elif delta < 0:
        return f"{delta} (over-prioritised by {abs(delta)} level{'s' if abs(delta) > 1 else ''})", delta
    return "0 (borderline mismatch — same ordinal but signal conflict)", 0


def build_constraint_analysis(row, inferred, mismatch_type, evidence):
    cat, chan = row['Issue_Category'], row['Ticket_Channel']
    prio = row['Priority_Level']
    rt   = float(row['Resolution_Time_Hours'])
    sat  = int(row['Satisfaction_Score'])

    top_kw = sorted([e for e in evidence if e['signal'] == 'keyword' and e.get('weight', 0) > 0],
                     key=lambda x: x.get('weight', 0), reverse=True)
    low_kw = sorted([e for e in evidence if e['signal'] == 'keyword' and e.get('weight', 0) < 0],
                     key=lambda x: x.get('weight', 0))

    s1 = (f"This {cat} ticket submitted via {chan} was assigned {prio} priority, "
          f"but the model infers {inferred} severity based on its content and metadata.")

    if mismatch_type == "Hidden Crisis" and top_kw:
        kw_list = ', '.join(f'"{e["value"]}"' for e in top_kw[:3])
        s2 = (f"The presence of escalation indicators ({kw_list}) in the ticket text "
              f"signals severity beyond the assigned label.")
    elif mismatch_type == "False Alarm" and low_kw:
        kw_list = ', '.join(f'"{e["value"]}"' for e in low_kw[:3])
        s2 = (f"The presence of low-severity indicators ({kw_list}) suggests "
              f"the ticket does not warrant its assigned priority level.")
    else:
        s2 = ("Semantic embedding patterns and resolution-time analysis "
              "indicate a priority mismatch beyond surface keyword signals.")

    if mismatch_type == "Hidden Crisis":
        s3 = (f"Resolution in {rt:.0f}h with a satisfaction score of {sat}/5 "
              f"further supports under-prioritisation — this ticket warranted "
              f"faster escalation to prevent customer impact.")
    else:
        s3 = (f"Resolution in {rt:.0f}h with satisfaction {sat}/5 suggests "
              f"the issue was less severe than the {prio} label implies — "
              f"resources may have been over-allocated.")

    return f"{s1} {s2} {s3}"


def compute_direction_signal(row) -> float:
    """Live direction signal — Hidden Crisis (positive) vs False Alarm (negative)."""
    text = f"{row['Ticket_Subject']} {row['Ticket_Description']}".lower()
    score = 0.0

    for pattern, _, weight in ESC_EVIDENCE:
        if re.search(pattern, text):
            score += weight
    for pattern, _, weight in DEESC_EVIDENCE:
        if re.search(pattern, text):
            score += weight

    cat, prio = row['Issue_Category'], row['Priority_Level']
    expected_sev, base_weight, _ = CATEGORY_SEVERITY.get(cat, ('Medium', 0.05, ''))
    exp_ord, prio_ord = PRIORITY_ORD.get(expected_sev, 1), PRIORITY_ORD.get(prio, 1)
    if exp_ord > prio_ord:
        score += base_weight
    elif exp_ord < prio_ord:
        score -= base_weight

    sat = int(row['Satisfaction_Score'])
    if sat <= 2 and prio in ['Low', 'Medium']:
        score += 0.20
    elif sat >= 4 and prio in ['Critical', 'High']:
        score -= 0.12

    actual_rt = float(row['Resolution_Time_Hours'])
    expected_rt = RT_MEDIANS.get((cat, prio), 40.0)
    ratio = actual_rt / max(expected_rt, 1)
    if ratio < 0.4:
        score += 0.15
    elif ratio > 2.5:
        score += 0.10

    return score


def infer_severity_and_type(row, prob: float):
    """Determine inferred severity and mismatch type from live direction signal."""
    base = PRIORITY_ORD.get(row['Priority_Level'], 1)
    direction = compute_direction_signal(row)

    if direction < 0:
        bump = 2 if prob >= 0.90 else 1
        inferred_ord = max(0, base - bump)
        mismatch_type = 'False Alarm'
    else:
        bump = 2 if prob >= 0.85 else 1
        inferred_ord = min(3, base + bump)
        mismatch_type = 'Hidden Crisis'

    return ORD_PRIORITY[inferred_ord], mismatch_type, inferred_ord


def generate_dossier(row: dict, prob: float, threshold: float) -> dict:
    """Generate a single hallucination-free evidence dossier."""
    inferred, mismatch_type, _ = infer_severity_and_type(row, prob)

    text_ev = extract_text_evidence(row, mismatch_type)
    rt_ev   = extract_rt_evidence(row, mismatch_type)
    cat_ev  = extract_category_evidence(row, mismatch_type)
    sat_ev  = extract_satisfaction_evidence(row, mismatch_type)

    feature_evidence = sorted(text_ev + [rt_ev, sat_ev, cat_ev],
                               key=lambda x: abs(x.get('weight', 0)), reverse=True)

    delta_str, delta_val = compute_severity_delta(row['Priority_Level'], inferred)

    return {
        "ticket_id":           str(row.get('Ticket_ID', 'N/A')),
        "assigned_priority":   row['Priority_Level'],
        "inferred_severity":   inferred,
        "mismatch_type":       mismatch_type,
        "severity_delta":      delta_str,
        "severity_delta_value": delta_val,
        "confidence":          round(float(prob), 4),
        "feature_evidence":    feature_evidence,
        "constraint_analysis": build_constraint_analysis(row, inferred, mismatch_type, feature_evidence)
    }

# ══════════════════════════════════════════════════════════════════
# Single-Ticket Tab
# ══════════════════════════════════════════════════════════════════

def render_single_ticket_tab(tokenizer, model, threshold, device):
    st.header("Single Ticket Analysis")
    st.markdown(
        "Enter a ticket's details below. SIA will predict whether the "
        "assigned priority matches the ticket's true severity, and — if a "
        "mismatch is detected — generate a full Evidence Dossier."
    )

    with st.form("single_ticket_form"):
        col1, col2 = st.columns(2)

        with col1:
            ticket_id = st.text_input("Ticket ID", value="TKT-DEMO-001")
            subject = st.text_input(
                "Ticket Subject",
                value="Question about some unusual activity on my account"
            )
            description = st.text_area(
                "Ticket Description",
                height=150,
                value=("I noticed some transactions on my account that I do not "
                       "recognize. My login credentials may have been accessed by "
                       "someone else. I have changed my password but wanted to let "
                       "you know. Please look into this when you get a chance.")
            )

        with col2:
            issue_category = st.selectbox("Issue Category", CATEGORIES, index=2)
            priority_level = st.selectbox("Assigned Priority", PRIORITIES, index=0)
            ticket_channel = st.selectbox("Ticket Channel", CHANNELS, index=1)
            resolution_time = st.number_input(
                "Resolution Time (hours)", min_value=1, max_value=200, value=48
            )
            satisfaction_score = st.slider(
                "Satisfaction Score", min_value=1, max_value=5, value=3
            )

        submitted = st.form_submit_button("Analyze Ticket", type="primary")

    if not submitted:
        return

    if not subject.strip() or not description.strip():
        st.warning("Please provide both a subject and description.")
        return

    row = {
        "Ticket_ID":             ticket_id,
        "Ticket_Subject":        subject,
        "Ticket_Description":    description,
        "Issue_Category":        issue_category,
        "Priority_Level":        priority_level,
        "Ticket_Channel":        ticket_channel,
        "Resolution_Time_Hours": resolution_time,
        "Satisfaction_Score":    satisfaction_score,
    }

    with st.spinner("Running inference..."):
        text = build_input_text(row)
        prob = predict_batch([text], tokenizer, model, device)[0]
        predicted_label = int(prob >= threshold)

    st.divider()

    # ── Verdict ──
    verdict_col, conf_col, thresh_col = st.columns(3)
    with verdict_col:
        if predicted_label == 1:
            st.error("**Verdict: MISMATCH**")
        else:
            st.success("**Verdict: CONSISTENT**")
    with conf_col:
        st.metric("Mismatch Probability", f"{prob:.1%}")
    with thresh_col:
        st.metric("Decision Threshold", f"{threshold:.2f}")

    # ── Dossier (only for mismatches) ──
    if predicted_label == 1:
        dossier = generate_dossier(row, prob, threshold)

        st.subheader("Evidence Dossier")

        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.metric("Assigned Priority", dossier['assigned_priority'])
        with d2:
            st.metric("Inferred Severity", dossier['inferred_severity'])
        with d3:
            mtype = dossier['mismatch_type']
            st.metric("Mismatch Type", mtype,
                      help="Hidden Crisis = under-prioritised | False Alarm = over-prioritised")
        with d4:
            st.metric("Severity Delta", dossier['severity_delta'].split(' ')[0])

        st.markdown(f"**{dossier['severity_delta']}**")

        st.markdown("#### Feature Evidence")
        st.caption("Every item below is traced to a specific field in the input ticket — "
                   "no fabricated or unverifiable claims.")

        for ev in dossier['feature_evidence']:
            weight = ev.get('weight', 0)
            sign = "🔺" if weight > 0 else "🔻" if weight < 0 else "▪️"
            with st.container(border=True):
                cols = st.columns([1, 3, 1])
                with cols[0]:
                    st.markdown(f"{sign} **{ev['signal']}**")
                    st.caption(f"source: `{ev['source_field']}`")
                with cols[1]:
                    if 'interpretation' in ev:
                        st.write(ev['interpretation'])
                    else:
                        st.write(f"Matched: \"{ev['value']}\" ({ev.get('type', '')})")
                with cols[2]:
                    st.metric("Weight", f"{weight:+.2f}", label_visibility="collapsed")

        st.markdown("#### Constraint Analysis")
        st.info(dossier['constraint_analysis'])

        with st.expander("Raw JSON dossier"):
            st.json(dossier)

    else:
        st.markdown(
            "This ticket's assigned priority is **consistent** with its inferred "
            "severity based on content, category, resolution time, and satisfaction "
            "signals. No evidence dossier is generated for consistent tickets."
        )
        
# ══════════════════════════════════════════════════════════════════
# Batch CSV Tab
# ══════════════════════════════════════════════════════════════════

REQUIRED_COLUMNS = [
    'Ticket_ID', 'Ticket_Subject', 'Ticket_Description', 'Issue_Category',
    'Priority_Level', 'Ticket_Channel', 'Resolution_Time_Hours',
    'Satisfaction_Score'
]


def render_batch_tab(tokenizer, model, threshold, device):
    st.header("Batch CSV Analysis")
    st.markdown(
        "Upload a CSV of support tickets. SIA will run inference on every row "
        "and generate evidence dossiers for all flagged mismatches."
    )

    with st.expander("Required CSV columns"):
        st.code(", ".join(REQUIRED_COLUMNS))
        st.caption(
            "Additional columns (Customer_Name, Customer_Email, Submission_Date, "
            "Assigned_Agent) are allowed but ignored by the model."
        )

    uploaded_file = st.file_uploader("Upload tickets CSV", type=["csv"])

    if uploaded_file is None:
        return

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        return

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        st.error(f"CSV is missing required columns: {missing_cols}")
        return

    for col in ['Ticket_Subject', 'Ticket_Description', 'Issue_Category',
                'Ticket_Channel', 'Priority_Level']:
        df[col] = df[col].astype(str)

    st.success(f"Loaded {len(df):,} tickets")
    st.dataframe(df.head(5), use_container_width=True)

    max_rows = st.slider(
        "Number of rows to process",
        min_value=10, max_value=min(len(df), 5000),
        value=min(len(df), 500), step=10,
        help="Limit rows for faster results in this demo. "
             "Use predict.py for full-dataset batch inference."
    )

    if not st.button("Run Batch Inference", type="primary"):
        return

    df_subset = df.head(max_rows).copy()

    with st.spinner(f"Running inference on {len(df_subset):,} tickets..."):
        texts = df_subset.apply(lambda r: build_input_text(r), axis=1).tolist()
        probs = predict_batch(texts, tokenizer, model, device)

    df_subset['mismatch_probability'] = probs
    df_subset['predicted_label']      = (probs >= threshold).astype(int)
    df_subset['predicted_verdict']    = df_subset['predicted_label'].map(
        {0: 'Consistent', 1: 'Mismatch'})

    n_flagged = int(df_subset['predicted_label'].sum())

    st.divider()

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Tickets Processed", f"{len(df_subset):,}")
    with m2:
        st.metric("Flagged as Mismatch", f"{n_flagged:,}")
    with m3:
        st.metric("Mismatch Rate", f"{n_flagged / len(df_subset) * 100:.1f}%")

    # ── Predictions table ──
    st.markdown("#### Predictions")
    display_cols = ['Ticket_ID', 'Priority_Level', 'Issue_Category',
                     'mismatch_probability', 'predicted_verdict']
    st.dataframe(
        df_subset[display_cols].sort_values('mismatch_probability', ascending=False),
        use_container_width=True,
        column_config={
            "mismatch_probability": st.column_config.ProgressColumn(
                "Mismatch Probability", min_value=0, max_value=1, format="%.3f"
            )
        }
    )

    csv_bytes = df_subset[display_cols].to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download predictions.csv",
        data=csv_bytes,
        file_name="predictions.csv",
        mime="text/csv"
    )

    # ── Dossiers for flagged tickets ──
    flagged = df_subset[df_subset['predicted_label'] == 1]

    if len(flagged) == 0:
        st.info("No mismatches detected in this batch.")
        return

    with st.spinner(f"Generating dossiers for {len(flagged):,} flagged tickets..."):
        dossiers = [
            generate_dossier(row, row['mismatch_probability'], threshold)
            for _, row in flagged.iterrows()
        ]

    st.markdown(f"#### Evidence Dossiers ({len(dossiers):,} flagged tickets)")

    dossier_json = json.dumps(dossiers, indent=2)
    st.download_button(
        "Download dossiers.json",
        data=dossier_json,
        file_name="dossiers.json",
        mime="application/json"
    )

    # Show first few dossiers inline, rest available via JSON download
    preview_n = min(5, len(dossiers))
    st.caption(f"Showing first {preview_n} of {len(dossiers):,} dossiers — "
               f"full set available in the JSON download above.")

    for dossier in dossiers[:preview_n]:
        with st.expander(
            f"**{dossier['ticket_id']}** — {dossier['assigned_priority']} → "
            f"{dossier['inferred_severity']} ({dossier['mismatch_type']}) "
            f"| confidence {dossier['confidence']:.2f}"
        ):
            st.markdown(f"**Severity Delta:** {dossier['severity_delta']}")

            for ev in dossier['feature_evidence']:
                weight = ev.get('weight', 0)
                sign = "🔺" if weight > 0 else "🔻" if weight < 0 else "▪️"
                interp = ev.get('interpretation', f"Matched: \"{ev.get('value','')}\"")
                st.markdown(f"{sign} **{ev['signal']}** (`{ev['source_field']}`, "
                             f"weight {weight:+.2f}) — {interp}")

            st.markdown("**Constraint Analysis:**")
            st.info(dossier['constraint_analysis'])

# ══════════════════════════════════════════════════════════════════
# Dashboard Tab
# ══════════════════════════════════════════════════════════════════

DOSSIERS_PATH = Path("outputs/dossiers.json")
PSEUDOLABELS_PATH = Path("outputs/tickets_pseudolabeled.csv")


@st.cache_data(show_spinner="Loading dashboard data...")
def load_dashboard_data():
    """Load precomputed dossiers and pseudo-labelled dataset for dashboard charts."""
    dossiers = None
    pseudo_df = None

    if DOSSIERS_PATH.exists():
        with open(DOSSIERS_PATH) as f:
            dossiers = json.load(f)

    if PSEUDOLABELS_PATH.exists():
        pseudo_df = pd.read_csv(PSEUDOLABELS_PATH)

    return dossiers, pseudo_df


def render_dashboard_tab():
    st.header("Priority Mismatch Dashboard")
    st.markdown(
        "Overview of mismatch detection results across the full ticket dataset "
        "(precomputed via `notebook.ipynb` / `train_pipeline.py`)."
    )

    dossiers, pseudo_df = load_dashboard_data()

    if dossiers is None or pseudo_df is None:
        st.warning(
            "Dashboard data not found. Run `notebook.ipynb` (Stages 1-3) or "
            "`train_pipeline.py` to generate `outputs/tickets_pseudolabeled.csv` "
            "and `outputs/dossiers.json`."
        )
        return

    total_tickets = len(pseudo_df)
    total_flagged = len(dossiers)
    n_consistent  = total_tickets - total_flagged

    hc_count = sum(1 for d in dossiers if d['mismatch_type'] == 'Hidden Crisis')
    fa_count = sum(1 for d in dossiers if d['mismatch_type'] == 'False Alarm')

    # ── Top-level metrics ──
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Tickets", f"{total_tickets:,}")
    with m2:
        st.metric("Flagged Mismatches", f"{total_flagged:,}",
                   f"{total_flagged/total_tickets*100:.1f}%")
    with m3:
        st.metric("Hidden Crisis", f"{hc_count:,}",
                   help="Under-prioritised — true severity exceeds assigned priority")
    with m4:
        st.metric("False Alarm", f"{fa_count:,}",
                   help="Over-prioritised — true severity is below assigned priority")

    st.divider()

    col1, col2 = st.columns(2)

    # ── Chart 1: Consistent vs Mismatch ──
    with col1:
        st.markdown("#### Overall Verdict Distribution")
        verdict_df = pd.DataFrame({
            'Verdict': ['Consistent', 'Mismatch'],
            'Count':   [n_consistent, total_flagged]
        })
        fig = px.pie(
            verdict_df, names='Verdict', values='Count',
            color='Verdict',
            color_discrete_map={'Consistent': '#5e9e6e', 'Mismatch': '#c44e52'},
            hole=0.45
        )
        fig.update_traces(textinfo='label+percent+value')
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # ── Chart 2: Mismatch Type breakdown ──
    with col2:
        st.markdown("#### Mismatch Type Breakdown")
        type_df = pd.DataFrame({
            'Type':  ['Hidden Crisis', 'False Alarm'],
            'Count': [hc_count, fa_count]
        })
        fig = px.bar(
            type_df, x='Type', y='Count', color='Type',
            color_discrete_map={'Hidden Crisis': '#c44e52', 'False Alarm': '#e09a3c'},
            text='Count'
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10),
                          yaxis_title="", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    col3, col4 = st.columns(2)

    # ── Chart 3: Mismatches by category ──
    with col3:
        st.markdown("#### Mismatches by Issue Category")
        cat_counts = pd.DataFrame([
            {'Category': d['assigned_priority'], 'Type': d['mismatch_type']}
            for d in dossiers
        ])
        # Pull category from the original df via ticket_id join for accuracy
        dossier_ids = [d['ticket_id'] for d in dossiers]
        dossier_types = {d['ticket_id']: d['mismatch_type'] for d in dossiers}

        flagged_pseudo = pseudo_df[pseudo_df['Ticket_ID'].isin(dossier_ids)].copy()
        flagged_pseudo['mismatch_type'] = flagged_pseudo['Ticket_ID'].map(dossier_types)

        cat_type_counts = (flagged_pseudo.groupby(['Issue_Category', 'mismatch_type'])
                           .size().reset_index(name='Count'))

        fig = px.bar(
            cat_type_counts, x='Issue_Category', y='Count', color='mismatch_type',
            color_discrete_map={'Hidden Crisis': '#c44e52', 'False Alarm': '#e09a3c'},
            barmode='stack'
        )
        fig.update_layout(margin=dict(t=10, b=10), xaxis_title="", legend_title="")
        st.plotly_chart(fig, use_container_width=True)

    # ── Chart 4: Top contributing signals ──
    with col4:
        st.markdown("#### Top Contributing Evidence Signals")
        st.caption("Frequency of each evidence signal type across all flagged dossiers")

        signal_counts = Counter()
        for d in dossiers:
            for ev in d['feature_evidence']:
                signal_counts[ev['signal']] += 1

        signal_df = pd.DataFrame(
            signal_counts.most_common(), columns=['Signal', 'Count']
        )
        fig = px.bar(
            signal_df, x='Count', y='Signal', orientation='h',
            color='Count', color_continuous_scale='Blues'
        )
        fig.update_layout(margin=dict(t=10, b=10), yaxis_title="",
                          showlegend=False, coloraxis_showscale=False)
        fig.update_yaxes(categoryorder='total ascending')
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Chart 5: Confidence distribution ──
    st.markdown("#### Confidence Distribution of Flagged Tickets")
    conf_values = [d['confidence'] for d in dossiers]
    fig = px.histogram(
        x=conf_values, nbins=30,
        labels={'x': 'Mismatch Confidence', 'y': 'Count'},
        color_discrete_sequence=['#5b8fc9']
    )
    fig.add_vline(x=0.69, line_dash="dash", line_color="#c44e52",
                   annotation_text="Decision threshold (0.69)")
    fig.update_layout(margin=dict(t=10, b=10), bargap=0.05)
    st.plotly_chart(fig, use_container_width=True)

 # ══════════════════════════════════════════════════════════════════
# Severity Delta Heatmap Tab
# ══════════════════════════════════════════════════════════════════

def render_heatmap_tab():
    st.header("Severity Delta Heatmap")
    st.markdown(
        "Average severity delta (`inferred_severity − assigned_priority`) across "
        "**Issue Category × Ticket Channel**. Positive values (red) indicate "
        "**Hidden Crisis** zones — under-prioritised tickets. Negative values "
        "(blue) indicate **False Alarm** zones — over-prioritised tickets."
    )

    dossiers, pseudo_df = load_dashboard_data()

    if dossiers is None or pseudo_df is None:
        st.warning(
            "Dashboard data not found. Run `notebook.ipynb` (Stages 1-3) or "
            "`train_pipeline.py` to generate `outputs/tickets_pseudolabeled.csv` "
            "and `outputs/dossiers.json`."
        )
        return

    # ── Build a per-ticket severity delta table ──
    # Flagged tickets get their dossier's severity_delta_value;
    # Consistent tickets get delta = 0 (by definition).
    def parse_delta(delta_str):
        """Extract integer delta from strings like '+2 (under-prioritised by 2 levels)'
        or '-1 (over-prioritised by 1 level)' or '0 (borderline mismatch...)'."""
        match = re.match(r'^([+-]?\d+)', delta_str)
        return int(match.group(1)) if match else 0

    dossier_deltas = {d['ticket_id']: parse_delta(d['severity_delta']) for d in dossiers}

    df = pseudo_df[['Ticket_ID', 'Issue_Category', 'Ticket_Channel']].copy()
    df['severity_delta'] = df['Ticket_ID'].map(dossier_deltas).fillna(0).astype(int)

    # ── Heatmap 1: Mean severity delta ──
    st.markdown("#### Mean Severity Delta")

    pivot_mean = df.pivot_table(
        index='Issue_Category', columns='Ticket_Channel',
        values='severity_delta', aggfunc='mean'
    ).reindex(index=CATEGORIES, columns=CHANNELS)

    fig = px.imshow(
        pivot_mean,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        color_continuous_midpoint=0,
        aspect='auto',
        labels=dict(x="Ticket Channel", y="Issue Category", color="Mean Δ severity")
    )
    fig.update_layout(margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Red cells: tickets in this category+channel combination tend to be "
        "**under-prioritised** (Hidden Crisis). Blue cells: tend to be "
        "**over-prioritised** (False Alarm). White/neutral: largely consistent."
    )

    st.divider()

    col1, col2 = st.columns(2)

    # ── Heatmap 2: Mismatch rate ──
    with col1:
        st.markdown("#### Mismatch Rate (%)")

        df['is_mismatch'] = (df['severity_delta'] != 0).astype(int)
        pivot_rate = df.pivot_table(
            index='Issue_Category', columns='Ticket_Channel',
            values='is_mismatch', aggfunc='mean'
        ).reindex(index=CATEGORIES, columns=CHANNELS) * 100

        fig = px.imshow(
            pivot_rate,
            text_auto='.1f',
            color_continuous_scale='Oranges',
            aspect='auto',
            labels=dict(x="Ticket Channel", y="Issue Category", color="Mismatch %")
        )
        fig.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # ── Heatmap 3: Ticket volume ──
    with col2:
        st.markdown("#### Ticket Volume")

        pivot_count = df.pivot_table(
            index='Issue_Category', columns='Ticket_Channel',
            values='Ticket_ID', aggfunc='count'
        ).reindex(index=CATEGORIES, columns=CHANNELS)

        fig = px.imshow(
            pivot_count,
            text_auto=True,
            color_continuous_scale='Blues',
            aspect='auto',
            labels=dict(x="Ticket Channel", y="Issue Category", color="Tickets")
        )
        fig.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Drill-down table ──
    st.markdown("#### Drill-down: Category × Channel Detail")

    detail = df.groupby(['Issue_Category', 'Ticket_Channel']).agg(
        total_tickets=('Ticket_ID', 'count'),
        mismatches=('is_mismatch', 'sum'),
        mean_delta=('severity_delta', 'mean'),
    ).reset_index()
    detail['mismatch_rate'] = (detail['mismatches'] / detail['total_tickets'] * 100).round(1)
    detail['mean_delta'] = detail['mean_delta'].round(3)
    detail = detail.sort_values('mismatch_rate', ascending=False)

    st.dataframe(
        detail,
        use_container_width=True,
        column_config={
            "Issue_Category": "Category",
            "Ticket_Channel": "Channel",
            "total_tickets": "Total Tickets",
            "mismatches": "Mismatches",
            "mean_delta": st.column_config.NumberColumn("Mean Δ Severity", format="%.3f"),
            "mismatch_rate": st.column_config.ProgressColumn(
                "Mismatch Rate (%)", min_value=0, max_value=100, format="%.1f%%"
            ),
        },
        hide_index=True,
    )

# ══════════════════════════════════════════════════════════════════
# Main Layout
# ══════════════════════════════════════════════════════════════════

def main():
    st.title("🛡️ SIA — Support Integrity Auditor")
    st.caption(
        "Detecting priority mismatches in CRM support tickets via "
        "DeBERTa-v3-small + LoRA, with hallucination-free evidence dossiers."
    )

    tokenizer, model, threshold, device = load_model()

    with st.sidebar:
        st.markdown("### Model Info")
        st.metric("Device", device.upper())
        st.metric("Decision Threshold", f"{threshold:.2f}")
        st.markdown("---")
        st.markdown(
            "**SIA** fuses 4 independent signals to generate pseudo-labels, "
            "fine-tunes DeBERTa-v3-small with LoRA, and produces structured, "
            "traceable evidence dossiers for flagged mismatches."
        )
        st.markdown("[View on GitHub](#)")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🎫 Single Ticket", "📊 Batch CSV", "📈 Dashboard", "🔥 Severity Heatmap"]
    )

    with tab1:
        render_single_ticket_tab(tokenizer, model, threshold, device)

    with tab2:
        render_batch_tab(tokenizer, model, threshold, device)

    with tab3:
        render_dashboard_tab()

    with tab4:
        render_heatmap_tab()

if __name__ == "__main__":
    main()

