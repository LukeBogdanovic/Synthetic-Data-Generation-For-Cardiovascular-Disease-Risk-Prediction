"""
compute_lead_stats.py
======================
Run this ONCE to recompute the global per-lead min/max values used during
training (per_lead_minmax_scaling). These are required for all six PTB-XL
evaluation scripts to scale real ECGs consistently with what the classifier
was trained on.

Adjust RAW_ECG_SOURCE below to point at whichever array holds the RAW
(unscaled) real ECG signals that were fed into per_lead_minmax_scaling
during training — likely the same array loaded from fine_tune_data.npy
before any scaling was applied.
"""

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

import numpy as np
from Final_models.WGAN.preprocessing_utils import per_lead_minmax_scaling

# ── Config ────────────────────────────────────────────────────────────────────

RAW_ECG_SOURCE = "fine_tune_data.npy"   # adjust if raw pre-scaling ECGs live elsewhere
OUT_PATH       = "lead_minmax_stats.npz"

# ── Load raw ECG data ─────────────────────────────────────────────────────────

raw_data = np.load(RAW_ECG_SOURCE, allow_pickle=True)

# fine_tune_data.npy is a list of (ecg, label) tuples per your CWGAN main() script
ecg_list = [item[0] for item in raw_data]
ecg_array = np.stack(ecg_list, axis=0)   # shape: (N, L, n_leads) — confirm this matches your data layout

print(f"Raw ECG array shape: {ecg_array.shape}")

if ecg_array.shape[-1] != 3:
    print(f"[Warning] Last dim is {ecg_array.shape[-1]}, expected 3 leads (III, V3, V5).")
    print("Check ecg_array.shape ordering before proceeding.")

# ── Compute global per-lead min/max exactly as done during training ──────────

_, lead_mins, lead_maxs = per_lead_minmax_scaling(ecg_array, feature_range=(-1, 1))

print(f"\nLead mins: {lead_mins}")
print(f"Lead maxs: {lead_maxs}")

np.savez(OUT_PATH, lead_mins=lead_mins, lead_maxs=lead_maxs)
print(f"\nSaved: {OUT_PATH}")