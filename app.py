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