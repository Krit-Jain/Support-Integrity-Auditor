#!/usr/bin/env python3
"""
predict.py — SIA (Support Integrity Auditor) Inference Script

Accepts a CSV of support tickets and outputs:
  1. predictions.csv  — every ticket with mismatch probability + verdict
  2. dossiers.json    — structured evidence dossiers for flagged tickets only

Usage:
    python predict.py --input data/tickets.csv --output outputs/

Required CSV columns:
    Ticket_ID, Ticket_Subject, Ticket_Description, Issue_Category,
    Priority_Level, Ticket_Channel, Resolution_Time_Hours, Satisfaction_Score
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel


# ──────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────

PRIORITY_ORD = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
ORD_PRIORITY = {0: 'Low', 1: 'Medium', 2: 'High', 3: 'Critical'}

REQUIRED_COLUMNS = [
    'Ticket_ID', 'Ticket_Subject', 'Ticket_Description', 'Issue_Category',
    'Priority_Level', 'Ticket_Channel', 'Resolution_Time_Hours',
    'Satisfaction_Score'
]

MODEL_NAME = "microsoft/deberta-v3-small"
MAX_LEN    = 256

# Evidence extraction patterns — identical to Stage 3
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


# ──────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────

def load_model(model_dir: Path, device: str):
    """Load the fine-tuned DeBERTa-v3-small + LoRA adapter and decision threshold."""
    best_dir = model_dir / "best"
    if not best_dir.exists():
        sys.exit(f"ERROR: model directory not found at {best_dir}. "
                 f"Run train_pipeline.py first.")

    tokenizer = AutoTokenizer.from_pretrained(str(best_dir))
    base = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True)
    model = PeftModel.from_pretrained(base, str(best_dir))
    model = model.float().to(device)
    model.eval()

    threshold_path = model_dir / "best_threshold.npy"
    threshold = float(np.load(str(threshold_path))[0]) if threshold_path.exists() else 0.5

    return tokenizer, model, threshold


# ──────────────────────────────────────────────────────────────────
# Input construction & inference
# ──────────────────────────────────────────────────────────────────

def rt_bin(rt_hours: float) -> str:
    if rt_hours <= 12:   return 'FAST'
    elif rt_hours <= 48: return 'MID'
    else:                return 'SLOW'


def build_input_text(row: pd.Series) -> str:
    """Hybrid input: text fields + structured metadata tokens."""
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


# ──────────────────────────────────────────────────────────────────
# Evidence extraction (Stage 3 logic — hallucination-free)
# ──────────────────────────────────────────────────────────────────

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
        return f"+{delta} (under-prioritised by {delta} level{'s' if delta > 1 else ''})"
    elif delta < 0:
        return f"{delta} (over-prioritised by {abs(delta)} level{'s' if abs(delta) > 1 else ''})"
    return "0 (borderline mismatch — same ordinal but signal conflict)"


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
    """
    Compute a direction score from live ticket signals (no pseudo-labels needed).
    Positive  -> Hidden Crisis direction (under-prioritised)
    Negative  -> False Alarm direction (over-prioritised)
    Mirrors Stage 1 Signals C+D logic, evaluated independently of mismatch_type.
    """
    text = f"{row['Ticket_Subject']} {row['Ticket_Description']}".lower()
    score = 0.0

    # Keyword direction: escalation -> +, de-escalation -> -
    for pattern, _, weight in ESC_EVIDENCE:
        if re.search(pattern, text):
            score += weight
    for pattern, _, weight in DEESC_EVIDENCE:
        if re.search(pattern, text):
            score += weight  # already negative

    # Category baseline direction
    cat, prio = row['Issue_Category'], row['Priority_Level']
    expected_sev, base_weight, _ = CATEGORY_SEVERITY.get(cat, ('Medium', 0.05, ''))
    exp_ord, prio_ord = PRIORITY_ORD.get(expected_sev, 1), PRIORITY_ORD.get(prio, 1)
    if exp_ord > prio_ord:
        score += base_weight
    elif exp_ord < prio_ord:
        score -= base_weight

    # Satisfaction direction
    sat = int(row['Satisfaction_Score'])
    if sat <= 2 and prio in ['Low', 'Medium']:
        score += 0.20   # unhappy customer on low-priority ticket -> HC
    elif sat >= 4 and prio in ['Critical', 'High']:
        score -= 0.12   # happy customer on high-priority ticket -> FA

    # RT direction
    actual_rt = float(row['Resolution_Time_Hours'])
    expected_rt = RT_MEDIANS.get((cat, prio), 40.0)
    ratio = actual_rt / max(expected_rt, 1)
    if ratio < 0.4:
        score += 0.15   # resolved much faster than expected -> HC
    elif ratio > 2.5:
        score += 0.10   # took much longer than expected -> HC (under-resourced)

    return score


def infer_severity_and_type(row, prob: float) -> tuple[str, str, int]:
    """
    Determine inferred severity and mismatch type using a live direction signal
    computed from ticket content, category baseline, satisfaction, and RT —
    independent of any pseudo-labels. Magnitude scales with model confidence.
    """
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


def generate_dossier(row, prob, threshold) -> dict:
    """Generate a single hallucination-free evidence dossier."""
    inferred, mismatch_type, _ = infer_severity_and_type(row, prob)

    text_ev = extract_text_evidence(row, mismatch_type)
    rt_ev   = extract_rt_evidence(row, mismatch_type)
    cat_ev  = extract_category_evidence(row, mismatch_type)
    sat_ev  = extract_satisfaction_evidence(row, mismatch_type)

    feature_evidence = sorted(text_ev + [rt_ev, sat_ev, cat_ev],
                               key=lambda x: abs(x.get('weight', 0)), reverse=True)

    return {
        "ticket_id":           str(row['Ticket_ID']),
        "assigned_priority":   row['Priority_Level'],
        "inferred_severity":   inferred,
        "mismatch_type":       mismatch_type,
        "severity_delta":      compute_severity_delta(row['Priority_Level'], inferred),
        "confidence":          round(float(prob), 4),
        "feature_evidence":    feature_evidence,
        "constraint_analysis": build_constraint_analysis(row, inferred, mismatch_type, feature_evidence)
    }


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SIA inference script")
    parser.add_argument("--input", required=True, help="Path to input CSV of tickets")
    parser.add_argument("--output", default="outputs/", help="Output directory")
    parser.add_argument("--model-dir", default="models/deberta_lora",
                         help="Path to trained model directory")
    parser.add_argument("--threshold", type=float, default=None,
                         help="Override decision threshold (default: use saved best_threshold.npy)")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_dir  = Path(args.output)
    model_dir   = Path(args.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        sys.exit(f"ERROR: input file not found: {input_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data
    df = pd.read_csv(input_path)
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        sys.exit(f"ERROR: input CSV is missing required columns: {missing_cols}")

    for col in ['Ticket_Subject', 'Ticket_Description', 'Issue_Category',
                'Ticket_Channel', 'Priority_Level']:
        df[col] = df[col].astype(str)

    print(f"Loaded {len(df):,} tickets from {input_path}")

    # Load model
    print(f"Loading model from {model_dir}/best ...")
    tokenizer, model, saved_threshold = load_model(model_dir, device)
    threshold = args.threshold if args.threshold is not None else saved_threshold
    print(f"✓ Model loaded | Decision threshold: {threshold:.2f}")

    # Build inputs and run inference
    texts = df.apply(build_input_text, axis=1).tolist()
    print(f"Running inference on {len(texts):,} tickets...")
    probs = predict_batch(texts, tokenizer, model, device, args.batch_size)

    df['mismatch_probability'] = probs
    df['predicted_label']      = (probs >= threshold).astype(int)
    df['predicted_verdict']    = df['predicted_label'].map(
        {0: 'Consistent', 1: 'Mismatch'})

    n_flagged = int(df['predicted_label'].sum())
    print(f"Flagged as mismatch: {n_flagged:,} / {len(df):,} "
          f"({n_flagged / len(df) * 100:.1f}%)")

    # Save predictions CSV
    pred_cols = ['Ticket_ID', 'Priority_Level', 'Issue_Category',
                  'mismatch_probability', 'predicted_label', 'predicted_verdict']
    pred_path = output_dir / "predictions.csv"
    df[pred_cols].to_csv(pred_path, index=False)
    print(f"✓ Predictions saved to {pred_path}")

    # Generate dossiers for flagged tickets
    flagged = df[df['predicted_label'] == 1]
    print(f"Generating dossiers for {len(flagged):,} flagged tickets...")

    dossiers = [generate_dossier(row, row['mismatch_probability'], threshold)
                for _, row in flagged.iterrows()]

    dossier_path = output_dir / "dossiers.json"
    with open(dossier_path, 'w') as f:
        json.dump(dossiers, f, indent=2)
    print(f"✓ {len(dossiers):,} dossiers saved to {dossier_path}")

    # Summary
    hc = sum(1 for d in dossiers if d['mismatch_type'] == 'Hidden Crisis')
    fa = sum(1 for d in dossiers if d['mismatch_type'] == 'False Alarm')
    print(f"\nSummary: Hidden Crisis={hc:,} | False Alarm={fa:,}")
    print("✓ Done")


if __name__ == "__main__":
    main()