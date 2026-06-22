import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

import torch
import numpy as np
import pandas as pd
import wfdb
from math import gcd
from scipy.signal import resample_poly
from torch.utils.data import TensorDataset, DataLoader

from Final_models.Classifier.classifier import Classifier
from Final_models.WGAN.preprocessing_utils import (
    setup_filter, bandpass_filter, extract_centered_segment_ptb
)

# ── Config ────────────────────────────────────────────────────────────────────

PTB_XL_PATH      = "data/pretrain"
CLASSIFIER_PTH   = "Final_models/Classifier/models/classifier18/best_model.pth"
LEAD_STATS_PATH  = "lead_minmax_stats.npz"   # from compute_lead_stats.py

N_SAMPLES      = 10
BATCH_SIZE     = 64
N_LEADS        = 3
SAMPLING_RATE  = 100   # PTB-XL source rate (lr files are 100Hz)
TARGET_HZ      = 128   # your model's expected rate
ECG_LENGTH     = 640   # 128Hz × 5s = 640
LEAD_INDICES   = [2, 8, 10]   # III, V3, V5

CLASS_NAMES    = {0: 'None', 1: 'MI', 2: 'Stroke', 3: 'Syncope'}
MI_LABEL       = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load training-derived lead min/max stats ──────────────────────────────────

stats     = np.load(LEAD_STATS_PATH)
LEAD_MINS = stats['lead_mins']
LEAD_MAXS = stats['lead_maxs']
print(f"Loaded lead stats — mins: {LEAD_MINS}, maxs: {LEAD_MAXS}")

b_filt, a_filt = setup_filter(lowcut=0.5, highcut=40, fs=TARGET_HZ, order=3)

# ── Step 1: Load PTB-XL & find MI records ────────────────────────────────────

ptbxl_db = pd.read_csv(os.path.join(PTB_XL_PATH, "ptbxl_database.csv"))
scp_df   = pd.read_csv(os.path.join(PTB_XL_PATH, "scp_statements.csv"), index_col=0)

mi_codes = scp_df[
    scp_df['diagnostic_class'].str.upper().str.contains('MI', na=False) |
    scp_df.index.str.upper().str.contains('MI', na=False)
].index.tolist()

def has_mi(scp_str):
    try:
        codes = eval(scp_str)
        return any(c in mi_codes for c in codes.keys())
    except Exception:
        return False

mi_records = ptbxl_db[ptbxl_db['scp_codes'].apply(has_mi)].reset_index(drop=True)
mi_sample  = mi_records.sample(n=N_SAMPLES, random_state=42).reset_index(drop=True)
print(f"Sampled {N_SAMPLES} MI records from PTB-XL")

# ── Step 2: Load & preprocess ECG signals ────────────────────────────────────

def load_ptbxl_ecg(row, ptb_path):
    fname  = os.path.join(ptb_path, row['filename_lr'])
    record = wfdb.rdrecord(fname)
    return record.p_signal.astype(np.float32)  # (samples, 12) in physical units (mV)

def preprocess_ecg(signal, lead_indices, target_length, lead_mins, lead_maxs,
                   src_hz=100, tgt_hz=128):
    sig = signal[:, lead_indices]               # (T, 3) — select III, V3, V5

    # Resample from src_hz to tgt_hz
    if src_hz != tgt_hz:
        g   = gcd(src_hz, tgt_hz)
        up  = tgt_hz // g
        dn  = src_hz // g
        sig = resample_poly(sig, up, dn, axis=0).astype(np.float32)

    # Bandpass filter (matches training preprocessing order)
    sig = bandpass_filter(sig, b_filt, a_filt).astype(np.float32)

    # Extract a 640-sample segment centered on an R-peak
    segment, found = extract_centered_segment_ptb(sig, fs=tgt_hz, segment_length=target_length)
    if not found:
        T = sig.shape[0]
        if T >= target_length:
            segment = sig[:target_length, :]
        else:
            pad = np.zeros((target_length - T, sig.shape[1]), dtype=np.float32)
            segment = np.concatenate([sig, pad], axis=0)

    # Global per-lead min-max scaling using TRAINING stats (not per-sample stats)
    segment = segment.astype(np.float32)
    scaled  = np.copy(segment)
    n_leads = segment.shape[1]
    for lead_idx in range(n_leads):
        denom = lead_maxs[lead_idx] - lead_mins[lead_idx]
        if denom == 0:
            denom = 1e-12
        scaled[:, lead_idx] = (segment[:, lead_idx] - lead_mins[lead_idx]) / denom
        scaled[:, lead_idx] = scaled[:, lead_idx] * 2 - 1   # to [-1, 1]

    return scaled  # (target_length, 3)

ecg_signals, meta_rows = [], []
for _, row in mi_sample.iterrows():
    try:
        signal   = load_ptbxl_ecg(row, PTB_XL_PATH)
        sig_proc = preprocess_ecg(signal, LEAD_INDICES, ECG_LENGTH, LEAD_MINS, LEAD_MAXS,
                                  src_hz=SAMPLING_RATE, tgt_hz=TARGET_HZ)
        ecg_signals.append(sig_proc)
        meta_rows.append(row)
    except Exception as e:
        print(f"  Skipping ecg_id={row['ecg_id']}: {e}")

if len(ecg_signals) == 0:
    raise RuntimeError(
        "No ECG signals loaded. Check PTB_XL_PATH.\n"
        f"  Example expected path: {os.path.join(PTB_XL_PATH, mi_sample.iloc[0]['filename_lr'])}"
    )

ecg_array  = np.stack(ecg_signals, axis=0)                                # (N, ECG_LENGTH, 3)
ecg_tensor = torch.tensor(ecg_array, dtype=torch.float32).permute(0, 2, 1)  # (N, 3, ECG_LENGTH)
n_loaded   = ecg_tensor.shape[0]
print(f"Loaded {n_loaded} ECG signals — shape: {ecg_tensor.shape}")

# ── Step 3: Build dummy CRFs ──────────────────────────────────────────────────

CRF_FEATURE_DIM = 7

MI_GROUP_MEANS = {
    'gender': 0.69,
    'age':    0.55,
    'weight': 0.48,
    'height': 0.50,
    'smoker': 0.31,
    'sbp':    0.52,
    'dbp':    0.47,
}

def build_dummy_crf(ptbxl_row, group_means, crf_dim):
    crf = np.array(list(group_means.values()), dtype=np.float32)
    if not pd.isna(ptbxl_row.get('sex', np.nan)):
        crf[0] = float(ptbxl_row['sex'])
    if not pd.isna(ptbxl_row.get('age', np.nan)):
        age_norm = (float(ptbxl_row['age']) - 46) / (92 - 46)
        crf[1]   = float(np.clip(age_norm, 0.0, 1.0))
    return crf

dummy_crfs = np.stack([
    build_dummy_crf(row, MI_GROUP_MEANS, CRF_FEATURE_DIM)
    for row in meta_rows
], axis=0)

crf_tensor = torch.tensor(dummy_crfs, dtype=torch.float32)
mi_labels  = torch.full((n_loaded,), MI_LABEL, dtype=torch.long)

print(f"Dummy CRF tensor shape: {crf_tensor.shape}")
print(f"\nDummy CRFs (per patient):")
crf_cols = list(MI_GROUP_MEANS.keys())
print(pd.DataFrame(dummy_crfs, columns=crf_cols).round(3).to_string(index=False))

# ── Step 4: Run through classifier ───────────────────────────────────────────

model = Classifier(num_leads=N_LEADS,
                   num_risk_factors=CRF_FEATURE_DIM,
                   num_classes=4).to(device)
model.load_state_dict(torch.load(CLASSIFIER_PTH, map_location=device))
model.eval()

dataset    = TensorDataset(ecg_tensor, crf_tensor, mi_labels)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

all_preds, all_probs = [], []
with torch.no_grad():
    for ecg_b, crf_b, _ in dataloader:
        logits = model(ecg_b.to(device), crf_b.to(device))
        probs  = torch.softmax(logits, dim=1)
        preds  = torch.argmax(probs, dim=1)
        all_preds.append(preds.cpu())
        all_probs.append(probs.cpu())

all_preds = torch.cat(all_preds).numpy()
all_probs = torch.cat(all_probs).numpy()

# ── Step 5: Results ───────────────────────────────────────────────────────────

meta_df = pd.DataFrame(meta_rows)[['ecg_id', 'patient_id', 'age', 'sex', 'scp_codes']].reset_index(drop=True)
results = meta_df.copy()
results['true_label']      = 'MI'
results['predicted_label'] = [CLASS_NAMES[p] for p in all_preds]
results['correct']         = results['predicted_label'] == 'MI'
results['prob_None']       = all_probs[:, 0]
results['prob_MI']         = all_probs[:, 1]
results['prob_Stroke']     = all_probs[:, 2]
results['prob_Syncope']    = all_probs[:, 3]
results['confidence']      = all_probs.max(axis=1)

print("\n── Results ──────────────────────────────────────────────────────")
print(results[['ecg_id', 'age', 'sex', 'predicted_label',
               'correct', 'prob_MI', 'confidence']].to_string(index=False))

correct = results['correct'].sum()
print(f"\nAccuracy on PTB-XL MI samples: {correct}/{n_loaded} ({correct/n_loaded*100:.1f}%)")
print("\nPrediction breakdown:")
print(results['predicted_label'].value_counts().to_string())

results.to_csv("ptbxl_mi_dummy_crf_results.csv", index=False)
print("\nSaved: ptbxl_mi_dummy_crf_results.csv")