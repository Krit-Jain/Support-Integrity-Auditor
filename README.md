# SIA — Support Integrity Auditor

**Detecting priority mismatches in CRM support tickets through multi-signal fusion, fine-tuned semantic classification, and hallucination-free evidence generation.**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-orange.svg)]()
[![DeBERTa-v3](https://img.shields.io/badge/model-DeBERTa--v3--small%20%2B%20LoRA-green.svg)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)]()

---

## Problem Statement

In enterprise-scale CRM ecosystems, manual ticket triage is riddled with agent fatigue bias, customer favoritism, and keyword anchoring. When critical issues are mislabeled as "Low" or trivial complaints are inflated to "Critical," Service Level Agreements (SLAs) are jeopardized and customer churn increases. Existing rule-based or keyword-matching systems fail to detect the nuanced discrepancies between a ticket's **true severity** and its **assigned priority**.

**SIA** is a self-supervised pipeline that:

1. **Generates pseudo-labels** for priority mismatches by fusing four independent signals — no ground-truth mismatch labels exist, so we bootstrap our own supervision
2. **Fine-tunes DeBERTa-v3-small with LoRA adapters** on the pseudo-labelled data to classify tickets as `Consistent` or `Mismatch`
3. **Generates structured, hallucination-free evidence dossiers** for every flagged ticket, with every claim traceable to a specific input field
4. **Survives adversarial attacks** designed to fool keyword-based triage systems

---

## Results at a Glance

| Metric | Result | Verification Threshold | Status |
|---|---|---|---|
| Binary Classification Accuracy | **92.05%** | ≥ 83% | ✅ Pass (+9.05pp) |
| Macro F1 Score | **0.8327** | ≥ 0.82 | ✅ Pass |
| Recall (Consistent) | **0.929** | ≥ 0.78 | ✅ Pass |
| Recall (Mismatch) | **0.856** | ≥ 0.78 | ✅ Pass |
| Adversarial Robustness | **7/10** | ≥ 7/10 | ✅ Bonus earned (+10%) |
| Dossier Hallucination Rate | **0 / 3,188** | 0 required | ✅ Pass |

---

## Architecture Overview

```
                       ┌──────────────────────────────────────────────┐
                       │   STAGE 1 — Pseudo-Label Generation          │
                       │   (Self-supervised, 4 fused signals)         │
                       ├──────────────────────────────────────────────┤
                       │  Signal A: Lexical severity + satisfaction   │
                       │  Signal B: Semantic embedding clustering     │
                       │  Signal C: Direct resolution-time mismatch   │
                       │  Signal D: Rule-based NLP (keywords/negation)│
                       │                                              │
                       │  → Logistic meta-learner fusion              │
                       │  → Binary pseudo-label: Mismatch | Consistent│
                       └───────────────────┬──────────────────────────┘
                                           │
                      ┌────────────────────▼───────────────────────────┐
                      │   STAGE 2 — Classifier Training                │
                      ├────────────────────────────────────────────────┤
                      │  DeBERTa-v3-small + LoRA (r=16, α=32)          │
                      │  Hybrid input: text + category + channel +     │
                      │                resolution-time bin + priority  │
                      │  Oversampling (1:3) + weighted CE loss         │
                      │  Native PyTorch training loop                  │
                      │  Threshold tuning on validation set            │
                      └───────────────────┬────────────────────────────┘
                                          │
                     ┌────────────────────▼────────────────────────────┐
                     │   STAGE 3 — Evidence Dossier Generation         │
                     ├─────────────────────────────────────────────────┤
                     │  Per flagged ticket: keyword evidence,          │
                     │  resolution-time analysis, category baseline,   │
                     │  satisfaction signal — all traced to source     │
                     │  fields. Zero LLM generation, zero hallucination│
                     └───────────────────┬─────────────────────────────┘
                                         │
                    ┌────────────────────▼─────────────────────────────┐
                    │   STAGE 4 — Adversarial Robustness Test          │
                    ├──────────────────────────────────────────────────┤
                    │  10 hand-crafted tickets with misleading surface │
                    │  language. Model relies on semantic context,     │
                    │  not keywords. Score: 7/10 → +10% bonus          │
                    └──────────────────────────────────────────────────┘
```