#!/usr/bin/env python3
"""
train_pipeline.py — SIA (Support Integrity Auditor) Training Pipeline

Reproduces the full pipeline from raw tickets to a trained, threshold-tuned
DeBERTa-v3-small + LoRA mismatch classifier.

Stages:
  1. Pseudo-label generation — 4 independent signals fused via logistic meta-learner
  2. DeBERTa-v3-small + LoRA fine-tuning (native PyTorch loop)

Usage:
    python train_pipeline.py --data data/customer_support_tickets.csv

Outputs:
    outputs/tickets_pseudolabeled.csv
    outputs/ablation_table.json
    models/deberta_lora/best/          (LoRA adapter + tokenizer)
    models/deberta_lora/best_threshold.npy
    models/deberta_lora/test_metrics.json
"""

import argparse
import json
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════
# Shared constants
# ══════════════════════════════════════════════════════════════════

PRIORITY_ORD = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
ORD_PRIORITY = {0: 'Low', 1: 'Medium', 2: 'High', 3: 'Critical'}

MODEL_NAME = "microsoft/deberta-v3-small"
MAX_LEN    = 256
SEED       = 42


# ══════════════════════════════════════════════════════════════════
# STAGE 1 — Pseudo-label generation
# ══════════════════════════════════════════════════════════════════

# ── Signal D: Rule-based NLP ──

ESC_PATTERNS = [
    (r'\bcannot\b',                          0.15),
    (r'\bunable\b',                          0.12),
    (r'\bcrash\w*\b',                        0.20),
    (r'\bnot (?:loading|working|syncing)\b', 0.18),
    (r'\bsuspicious\b',                      0.20),
    (r'\bphish\w*\b',                        0.30),
    (r'\bstolen\b',                          0.30),
    (r'\bfraud\w*\b',                        0.30),
    (r'\bhack\w*\b',                         0.25),
    (r'\bdata.*(?:loss|breach|leak)\b',      0.30),
    (r'\b24 hours?\b',                       0.12),
    (r'\bimmediately\b',                     0.15),
    (r'\blocke?d? out\b',                    0.20),
    (r'\baccount.*(?:suspend|comprom)\w*\b', 0.25),
    (r'\bpayment.*fail\w*\b',                0.18),
    (r'\bdashboard.*not\b',                  0.18),
    (r'\bsettings.*crash\w*\b',              0.20),
    (r'\bsync.*fail\w*\b',                   0.18),
]

DEESC_PATTERNS_D = [
    r'\bwhere is\b', r'\bhow do i\b', r'\bquestion\b',
    r'\broadmap\b',  r'\bheadquarters\b', r'\bupgrade.*plan\b',
    r'\bfeature.*request\b', r'\bhours of operation\b', r'\bwhat is\b',
]

CATEGORY_WEIGHTS_D = {
    'Fraud': 1.0, 'Technical': 0.70, 'Account': 0.50,
    'Billing': 0.40, 'General Inquiry': 0.10
}


def compute_signal_d(row) -> float:
    """Rule-based NLP severity score [0,1] from escalation/de-escalation keywords."""
    text = f"{row['Ticket_Subject']} {row['Ticket_Description']}".lower()

    esc_score = min(0.80, sum(w for p, w in ESC_PATTERNS if re.search(p, text)))
    deesc_pen = 0.10 * sum(1 for p in DEESC_PATTERNS_D if re.search(p, text))

    words = text.split()
    neg_words = ['not', 'no', "n't", 'never', 'unable', 'cannot', 'cant']
    neg_density = sum(1 for w in words if w in neg_words) / max(len(words), 1)

    cat_w = CATEGORY_WEIGHTS_D.get(row['Issue_Category'], 0.30)

    score = esc_score * 0.50 + neg_density * 1.50 + cat_w * 0.30 - deesc_pen
    return float(np.clip(score, 0.0, 1.0))


# ── Signal C: Direct RT mismatch score ──

def compute_signal_c(row, rt_lookup, global_rt_median) -> float:
    """Direct comparison of actual vs expected resolution time for assigned priority."""
    actual_rt = float(row['Resolution_Time_Hours'])
    prio      = row['Priority_Level']
    prio_w    = PRIORITY_ORD[prio] / 3.0

    expected_rt = rt_lookup.get((row['Issue_Category'], prio),
                                 global_rt_median.get(prio, 40.0))

    too_slow = max(0.0, actual_rt - expected_rt) / max(expected_rt, 1.0)
    too_fast = max(0.0, expected_rt - actual_rt) / max(expected_rt, 1.0)

    score = prio_w * too_slow + (1.0 - prio_w) * too_fast
    return float(np.clip(score * 0.5, 0, 1))


# ── Signal A: Lexical severity + satisfaction inversion ──

HIGH_SEV_ANCHORS = [
    "cannot access account locked credentials stolen phishing fraud hacked",
    "system down crash not loading broken urgent data loss payment failed",
    "unauthorized suspicious transaction security breach critical error",
    "refund issue billing charge wrong account suspended terminate",
    "dashboard broken sync failed settings crash api down outage"
]
LOW_SEV_ANCHORS = [
    "how do i where is question general inquiry feature request roadmap",
    "headquarters office hours upgrade plan update information",
    "recommendation suggestion feedback compliment looking for"
]
_HIGH_WORDS = set(re.findall(r'\b\w+\b', ' '.join(HIGH_SEV_ANCHORS).lower()))
_LOW_WORDS  = set(re.findall(r'\b\w+\b', ' '.join(LOW_SEV_ANCHORS).lower()))

CATEGORY_BASE_A = {
    'Fraud': 0.85, 'Technical': 0.55, 'Account': 0.40,
    'Billing': 0.35, 'General Inquiry': 0.10,
}


def compute_signal_a(row) -> float:
    """Deterministic lexical severity + satisfaction inversion + category prior."""
    text  = f"{row['Ticket_Subject']} {row['Ticket_Description']}".lower()
    words = set(re.findall(r'\b\w+\b', text))

    high_overlap = len(words & _HIGH_WORDS) / max(len(_HIGH_WORDS), 1)
    low_overlap  = len(words & _LOW_WORDS)  / max(len(_LOW_WORDS), 1)
    lex_score    = float(np.clip((high_overlap * 3.0 - low_overlap * 1.5) * 4.0, 0, 1))

    sat_score = (5 - int(row['Satisfaction_Score'])) / 4.0
    cat_score = CATEGORY_BASE_A.get(row['Issue_Category'], 0.30)

    return float(np.clip(0.50 * lex_score + 0.30 * sat_score + 0.20 * cat_score, 0, 1))


# ── Signal B: Embedding-based semantic clustering (Windows-safe with fallbacks) ──

def compute_signal_b(df: pd.DataFrame, output_dir: Path) -> tuple[np.ndarray, str]:
    """
    Semantic clustering severity score. Falls back gracefully:
      sentence-transformers -> TF-IDF+SVD
      UMAP                   -> PCA
      HDBSCAN                -> KMeans
    Returns (signal_b array, method description string).
    """
    df['text_combined'] = (
        df['Ticket_Subject'].fillna('') + '. ' +
        df['Ticket_Description'].fillna('') + '. Category: ' +
        df['Issue_Category'].fillna('') + '. Channel: ' +
        df['Ticket_Channel'].fillna('')
    )

    def _import_ok(name):
        try:
            __import__(name)
            return True
        except Exception:
            return False

    torch_ok = _import_ok("torch")
    sbert_ok = _import_ok("sentence_transformers") and torch_ok
    umap_ok  = _import_ok("umap")
    hdb_ok   = _import_ok("hdbscan")

    print(f"  sentence-transformers: {'✓' if sbert_ok else '✗ (fallback: TF-IDF+SVD)'}")
    print(f"  umap-learn           : {'✓' if umap_ok else '✗ (fallback: PCA)'}")
    print(f"  hdbscan              : {'✓' if hdb_ok else '✗ (fallback: KMeans)'}")

    if sbert_ok:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = embedder.encode(
            df['text_combined'].tolist(), batch_size=64,
            show_progress_bar=True, normalize_embeddings=True)
        method = "sentence-transformers"
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import normalize
        tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1, 2),
                                 sublinear_tf=True, min_df=3)
        X_tfidf = tfidf.fit_transform(df['text_combined'])
        svd = TruncatedSVD(n_components=100, random_state=SEED)
        embeddings = normalize(svd.fit_transform(X_tfidf))
        method = "tfidf-svd"

    if umap_ok:
        import umap
        reducer = umap.UMAP(n_components=10, n_neighbors=20, min_dist=0.1,
                             metric='cosine', random_state=SEED, verbose=False)
        embeddings_reduced = reducer.fit_transform(embeddings)
    else:
        from sklearn.decomposition import TruncatedSVD as _SVD
        embeddings_reduced = _SVD(n_components=min(10, embeddings.shape[1] - 1),
                                   random_state=SEED).fit_transform(embeddings)

    if hdb_ok:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10,
                                     metric='euclidean', prediction_data=True)
        cluster_labels = clusterer.fit_predict(embeddings_reduced)
    else:
        from sklearn.cluster import KMeans
        cluster_labels = KMeans(n_clusters=12, random_state=SEED, n_init=10).fit_predict(embeddings_reduced)

    df['cluster_id'] = cluster_labels

    cluster_priority_map = {}
    for cid in set(cluster_labels):
        if cid == -1:
            continue
        mask = df['cluster_id'] == cid
        cluster_priority_map[cid] = df.loc[mask, 'priority_ord'].mean()

    if cluster_priority_map:
        min_p, max_p = min(cluster_priority_map.values()), max(cluster_priority_map.values())
        rng = max_p - min_p + 1e-9
        signal_b = df['cluster_id'].map(
            lambda cid: float((cluster_priority_map.get(cid, min_p) - min_p) / rng)
            if cid != -1 else 0.5
        ).values
    else:
        signal_b = np.full(len(df), 0.5)

    signal_b = np.clip(signal_b, 0, 1)

    np.save(str(output_dir / "embeddings_reduced.npy"), embeddings_reduced)
    np.save(str(output_dir / "cluster_labels.npy"), cluster_labels)

    return signal_b, method


# ── Fusion + ablation ──

def fuse_signals_and_label(df: pd.DataFrame, output_dir: Path, model_dir: Path) -> pd.DataFrame:
    """Fuse all 4 signals via logistic meta-learner, compute ablation table, assign pseudo-labels."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import cohen_kappa_score
    from scipy import stats as scipy_stats

    SIGNAL_COLS = ['signal_a', 'signal_b', 'signal_c', 'signal_d']
    SIGNAL_NAMES = {
        'signal_a': 'Lexical severity + satisfaction',
        'signal_b': 'Semantic clustering (embeddings)',
        'signal_c': 'RT mismatch (direct)',
        'signal_d': 'Rule-based NLP',
    }

    print("\n── Pairwise signal agreement (Cohen's κ) ──")
    for i, s1 in enumerate(SIGNAL_COLS):
        for s2 in SIGNAL_COLS[i+1:]:
            b1 = (df[s1] >= 0.5).astype(int)
            b2 = (df[s2] >= 0.5).astype(int)
            kappa = cohen_kappa_score(b1, b2)
            agree = (b1 == b2).mean() * 100
            print(f"  {SIGNAL_NAMES[s1][:28]:28s} x {SIGNAL_NAMES[s2][:24]:24s} "
                  f"κ={kappa:.3f}  agree={agree:.1f}%")

    print("\n── Ablation table ──")
    ablation_results = {}
    for col in SIGNAL_COLS:
        inferred = df[col].apply(lambda s:
            3 if s >= 0.75 else 2 if s >= 0.50 else 1 if s >= 0.25 else 0)
        acc1 = (abs(inferred - df['priority_ord']) <= 1).mean()
        rho, _ = scipy_stats.spearmanr(df[col], df['priority_ord'])
        print(f"  {SIGNAL_NAMES[col]:<32} rho={rho:.3f}  acc@1={acc1*100:.1f}%")
        ablation_results[col] = {'rho': rho, 'acc1': acc1}

    X = df[SIGNAL_COLS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    df['true_severe'] = (df['priority_ord'] >= 2).astype(int)
    y = df['true_severe'].values

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    lr = LogisticRegression(C=1.0, class_weight='balanced', random_state=SEED, max_iter=500)
    cv_acc = cross_val_score(lr, X_scaled, y, cv=cv, scoring='accuracy')
    cv_f1  = cross_val_score(lr, X_scaled, y, cv=cv, scoring='f1_macro')
    print(f"\n── Learned fusion (logistic meta-learner) ──")
    print(f"  5-fold CV Accuracy : {cv_acc.mean():.3f} ± {cv_acc.std():.3f}")
    print(f"  5-fold CV Macro-F1 : {cv_f1.mean():.3f} ± {cv_f1.std():.3f}")

    lr.fit(X_scaled, y)
    df['severity_score_fused'] = lr.predict_proba(X_scaled)[:, 1]

    weights = dict(zip(SIGNAL_COLS, np.abs(lr.coef_[0])))
    total_w = sum(weights.values())
    print("\n  Signal weights (normalized):")
    for col, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"    {SIGNAL_NAMES[col]:<32}: {w/total_w*100:.1f}%")

    # ── Pseudo-label assignment ──
    def assign_inferred_severity(score):
        if score >= 0.75: return 'Critical'
        elif score >= 0.50: return 'High'
        elif score >= 0.25: return 'Medium'
        return 'Low'

    df['inferred_severity'] = df['severity_score_fused'].apply(assign_inferred_severity)
    df['inferred_ord']      = df['inferred_severity'].map(PRIORITY_ORD)
    df['severity_delta']    = df['inferred_ord'] - df['priority_ord']
    df['label']             = (abs(df['severity_delta']) >= 2).astype(int)

    def mismatch_type(row):
        if row['label'] == 0:
            return 'Consistent'
        return 'Hidden Crisis' if row['severity_delta'] >= 2 else 'False Alarm'

    df['mismatch_type'] = df.apply(mismatch_type, axis=1)

    print(f"\n── Pseudo-label distribution ──")
    print(df['label'].value_counts().rename({0: 'Consistent', 1: 'Mismatch'}).to_string())
    print(df['mismatch_type'].value_counts().to_string())
    pct = df['label'].mean() * 100
    print(f"Mismatch rate: {pct:.1f}%")

    # Save ablation table for README
    ablation_json = {
        SIGNAL_NAMES[c]: {
            'spearman_rho': round(ablation_results[c]['rho'], 4),
            'acc_within_1': round(ablation_results[c]['acc1'] * 100, 2),
            'weight_pct':   round(weights[c] / total_w * 100, 1)
        }
        for c in SIGNAL_COLS
    }
    with open(output_dir / "ablation_table.json", "w") as f:
        json.dump(ablation_json, f, indent=2)

    import pickle
    with open(model_dir / "fusion_model.pkl", "wb") as f:
        pickle.dump({'lr': lr, 'scaler': scaler, 'signal_cols': SIGNAL_COLS}, f)

    return df


def run_stage1(data_path: Path, output_dir: Path, model_dir: Path) -> pd.DataFrame:
    """Stage 1 — full pseudo-label generation pipeline."""
    print("=" * 60)
    print("STAGE 1 — Pseudo-label generation")
    print("=" * 60)

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} tickets")

    df['priority_ord'] = df['Priority_Level'].map(PRIORITY_ORD)

    print("\nComputing Signal D (rule-based NLP)...")
    df['signal_d'] = df.apply(compute_signal_d, axis=1)

    print("Computing Signal C (direct RT mismatch)...")
    rt_lookup = (df.groupby(['Issue_Category', 'Priority_Level'])['Resolution_Time_Hours']
                   .median().to_dict())
    global_rt_median = df.groupby('Priority_Level')['Resolution_Time_Hours'].median().to_dict()
    df['signal_c'] = df.apply(lambda r: compute_signal_c(r, rt_lookup, global_rt_median), axis=1)

    print("Computing Signal A (lexical severity + satisfaction)...")
    df['signal_a'] = df.apply(compute_signal_a, axis=1)

    print("\nComputing Signal B (semantic clustering)...")
    df['signal_b'], method = compute_signal_b(df, output_dir)
    print(f"  Method used: {method}")

    df = fuse_signals_and_label(df, output_dir, model_dir)

    out_path = output_dir / "tickets_pseudolabeled.csv"
    df.to_csv(out_path, index=False)
    print(f"\n✓ Pseudo-labelled dataset saved to {out_path}")

    return df


# ══════════════════════════════════════════════════════════════════
# STAGE 2 — DeBERTa-v3-small + LoRA training
# ══════════════════════════════════════════════════════════════════

def rt_bin(rt_hours: float) -> str:
    if rt_hours <= 12:   return 'FAST'
    elif rt_hours <= 48: return 'MID'
    return 'SLOW'


def build_input_text(row) -> str:
    return (f"{row['Ticket_Subject']} [SEP] "
            f"{row['Ticket_Description']} [SEP] "
            f"category:{row['Issue_Category']} "
            f"channel:{row['Ticket_Channel']} "
            f"rt:{rt_bin(float(row['Resolution_Time_Hours']))} "
            f"priority:{row['Priority_Level']}")


class TicketDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.enc = tokenizer(list(texts), truncation=True,
                              padding='max_length', max_length=max_len,
                              return_tensors='pt')
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.enc.items()}
        item['labels'] = self.labels[idx]
        return item


def evaluate(model, loader, device, threshold=0.5):
    from sklearn.metrics import accuracy_score, f1_score
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels']
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    all_probs, all_labels = np.array(all_probs), np.array(all_labels)
    preds = (all_probs >= threshold).astype(int)
    acc = accuracy_score(all_labels, preds)
    f1  = f1_score(all_labels, preds, average='macro', zero_division=0)
    f1c = f1_score(all_labels, preds, average=None, zero_division=0)
    return acc, f1, f1c, all_probs, all_labels


def run_stage2(df: pd.DataFrame, model_dir: Path, epochs: int, batch_size: int):
    """Stage 2 — DeBERTa-v3-small + LoRA fine-tuning, native PyTorch loop."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (accuracy_score, f1_score,
                                  classification_report, confusion_matrix)
    from sklearn.utils import resample
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                               get_cosine_schedule_with_warmup)
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel

    print("\n" + "=" * 60)
    print("STAGE 2 — DeBERTa-v3-small + LoRA fine-tuning")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}" + (f" | {torch.cuda.get_device_name(0)}" if device == "cuda" else ""))

    for col in ['Ticket_Subject', 'Ticket_Description', 'Issue_Category',
                'Ticket_Channel', 'Priority_Level']:
        df[col] = df[col].astype(str)

    df['input_text'] = df.apply(build_input_text, axis=1)
    X = np.array(df['input_text'].tolist())
    y = np.array(df['label'].tolist(), dtype=int)

    print(f"Loaded {len(df):,} | Mismatch: {y.mean()*100:.1f}%")

    # Split 80/10/10
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.10, stratify=y, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.111, stratify=y_tv, random_state=SEED)
    print(f"Train:{len(X_train):,} Val:{len(X_val):,} Test:{len(X_test):,}")

    # Oversample minority 1:3
    X_maj, y_maj = X_train[y_train == 0], y_train[y_train == 0]
    X_min, y_min = X_train[y_train == 1], y_train[y_train == 1]
    target_min = len(X_maj) // 3
    X_min_up = resample(X_min, replace=True, n_samples=target_min, random_state=SEED)
    y_min_up = np.ones(target_min, dtype=int)
    X_train_bal = np.concatenate([X_maj, X_min_up])
    y_train_bal = np.concatenate([y_maj, y_min_up])
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(X_train_bal))
    X_train_bal, y_train_bal = X_train_bal[idx], y_train_bal[idx]
    print(f"Train bal: {len(X_train_bal):,} | "
          f"Consistent:{(y_train_bal==0).sum():,} Mismatch:{(y_train_bal==1).sum():,}")

    # Tokenizer + datasets
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = TicketDataset(X_train_bal, y_train_bal, tokenizer)
    val_ds   = TicketDataset(X_val,       y_val,       tokenizer)
    test_ds  = TicketDataset(X_test,      y_test,      tokenizer)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=32, shuffle=False)
    test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=32, shuffle=False)
    print(f"✓ DataLoaders ready | Train batches: {len(train_loader)}")

    # Model + LoRA
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, lora_dropout=0.1,
        bias="none",
        target_modules=["query_proj", "key_proj", "value_proj", "out_proj"],
        modules_to_save=["classifier", "pooler"]
    )
    model = get_peft_model(base_model, lora_config)
    model.to(device)
    model = model.float()  # CRITICAL: float32 — bf16 causes nan gradients in native loop

    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable: {trainable:,} ({100*trainable/total:.2f}%)")

    # Weighted loss
    n_con, n_mis = (y_train_bal == 0).sum(), (y_train_bal == 1).sum()
    w0, w1 = len(y_train_bal) / (2*n_con), len(y_train_bal) / (2*n_mis)
    class_weights = torch.tensor([w0, w1], dtype=torch.float).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    print(f"Loss weights — Consistent:{w0:.3f}  Mismatch:{w1:.3f}")

    # Optimizer + scheduler
    GRAD_ACCUM = 4
    LR = 2e-5
    total_steps  = (len(train_loader) // GRAD_ACCUM) * epochs
    warmup_steps = int(0.06 * total_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    print(f"\nTraining: epochs={epochs} grad_accum={GRAD_ACCUM} lr={LR} "
          f"total_steps={total_steps} warmup={warmup_steps}")

    # Training loop
    print(f"\n{'Epoch':>5} {'TrainLoss':>10} {'ValAcc':>8} {'MacroF1':>9} {'F1_Con':>8} {'F1_Mis':>8}")
    print("-" * 60)

    best_val_f1, best_epoch = 0.0, 0
    optimizer.zero_grad()
    best_dir = model_dir / "best"

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels) / GRAD_ACCUM
            loss.backward()
            total_loss += loss.item() * GRAD_ACCUM

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_loss = total_loss / len(train_loader)
        val_acc, val_f1, val_f1c, _, _ = evaluate(model, val_loader, device)

        print(f"{epoch:>5} {avg_loss:>10.4f} {val_acc*100:>7.2f}% "
              f"{val_f1:>9.4f} {val_f1c[0]:>8.4f} {val_f1c[1]:>8.4f}")

        if val_f1 > best_val_f1:
            best_val_f1, best_epoch = val_f1, epoch
            model.save_pretrained(str(best_dir))
            tokenizer.save_pretrained(str(best_dir))

    print(f"\nBest epoch: {best_epoch} | Best val Macro-F1: {best_val_f1:.4f}")

    # Reload best checkpoint
    print("\nLoading best checkpoint for final evaluation...")
    best_base = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True)
    best_model = PeftModel.from_pretrained(best_base, str(best_dir))
    best_model = best_model.float().to(device)
    best_model.eval()

    # Threshold tuning on val
    _, _, _, val_probs, val_labels = evaluate(best_model, val_loader, device)
    best_t, best_f1_t = 0.5, 0.0
    for t in np.arange(0.15, 0.75, 0.01):
        p = (val_probs >= t).astype(int)
        f = f1_score(val_labels, p, average='macro', zero_division=0)
        if f > best_f1_t:
            best_f1_t, best_t = f, t
    print(f"Best threshold: {best_t:.2f} | Val Macro-F1: {best_f1_t:.4f}")

    # Final test evaluation
    _, _, _, test_probs, test_labels_arr = evaluate(best_model, test_loader, device)
    test_preds = (test_probs >= best_t).astype(int)

    acc    = accuracy_score(test_labels_arr, test_preds)
    f1_mac = f1_score(test_labels_arr, test_preds, average='macro', zero_division=0)
    report = classification_report(test_labels_arr, test_preds,
                                    target_names=['Consistent', 'Mismatch'], zero_division=0)
    cm = confusion_matrix(test_labels_arr, test_preds)
    per_class_recall = cm.diagonal() / cm.sum(axis=1)

    print(f"\n{'='*60}\nFINAL TEST RESULTS (threshold={best_t:.2f})\n{'='*60}")
    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"  Macro F1 : {f1_mac:.4f}")
    print(report)
    print(f"Confusion Matrix:\n{cm}")

    print(f"\n── Verification criteria ──")
    print(f"  Accuracy  : {acc*100:.2f}%  {'✓ PASS' if acc>=0.83 else '✗ FAIL'} (≥83%)")
    print(f"  Macro F1  : {f1_mac:.4f}  {'✓ PASS' if f1_mac>=0.82 else '✗ FAIL'} (≥0.82)")
    print(f"  Recall C  : {per_class_recall[0]:.4f}  {'✓ PASS' if per_class_recall[0]>=0.78 else '✗ FAIL'} (≥0.78)")
    print(f"  Recall M  : {per_class_recall[1]:.4f}  {'✓ PASS' if per_class_recall[1]>=0.78 else '✗ FAIL'} (≥0.78)")

    # Save final artifacts
    model.save_pretrained(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    np.save(str(model_dir / "best_threshold.npy"), np.array([best_t]))

    metrics_out = {
        "accuracy":          round(acc, 4),
        "macro_f1":          round(f1_mac, 4),
        "recall_consistent": round(float(per_class_recall[0]), 4),
        "recall_mismatch":   round(float(per_class_recall[1]), 4),
        "best_threshold":    round(float(best_t), 2),
        "confusion_matrix":  cm.tolist(),
        "model":             MODEL_NAME,
        "lora_r":            16,
        "lora_alpha":        32,
        "best_epoch":        best_epoch,
        "epochs_trained":    epochs,
    }
    with open(model_dir / "test_metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    print(f"\n✓ Model saved to {model_dir}")
    print(f"✓ Metrics saved to {model_dir}/test_metrics.json")


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SIA training pipeline")
    parser.add_argument("--data", required=True, help="Path to raw tickets CSV")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--model-dir", default="models/deberta_lora", help="Model output directory")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--skip-stage1", action="store_true",
                         help="Skip pseudo-labeling, load existing tickets_pseudolabeled.csv")
    args = parser.parse_args()

    data_path  = Path(args.data)
    output_dir = Path(args.output_dir)
    model_dir  = Path(args.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        sys.exit(f"ERROR: data file not found: {data_path}")

    if args.skip_stage1:
        pseudo_path = output_dir / "tickets_pseudolabeled.csv"
        if not pseudo_path.exists():
            sys.exit(f"ERROR: --skip-stage1 set but {pseudo_path} not found")
        print(f"Skipping Stage 1, loading {pseudo_path}")
        df = pd.read_csv(pseudo_path)
        df = df.convert_dtypes(dtype_backend="numpy_nullable")
    else:
        df = run_stage1(data_path, output_dir, model_dir)
        df = df.convert_dtypes(dtype_backend="numpy_nullable")

    run_stage2(df, model_dir, args.epochs, args.batch_size)

    print("\n✓ train_pipeline.py complete")


if __name__ == "__main__":
    main()