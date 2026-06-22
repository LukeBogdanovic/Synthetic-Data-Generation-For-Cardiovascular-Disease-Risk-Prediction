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

from Final_models.Classifier.classifier_ecg import Classifier
from Final_models.WGAN.preprocessing_utils import (
    setup_filter, bandpass_filter, extract_centered_segment_ptb
)

# ── Config ────────────────────────────────────────────────────────────────────

PTB_XL_PATH      = "data/pretrain"
CLASSIFIER_PTH   = "Final_models/Classifier/models/classifier_ecg_only_0/best_model.pth"   # update N to match your training run
LEAD_STATS_PATH  = "lead_minmax_stats.npz"

N_SAMPLES      = 10
BATCH_SIZE     = 64
N_LEADS        = 3
SAMPLING_RATE  = 100
TARGET_HZ      = 128
ECG_LENGTH     = 640
LEAD_INDICES   = [2, 8, 10]   # III, V3, V5

CLASS_NAMES    = {0: 'None', 1: 'MI', 2: 'Stroke'}
NUM_CLASSES    = 3
STROKE_LABEL   = 2   # AFib used as a proxy for Stroke

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load training-derived lead min/max stats ──────────────────────────────────

stats     = np.load(LEAD_STATS_PATH)
LEAD_MINS = stats['lead_mins']
LEAD_MAXS = stats['lead_maxs']
print(f"Loaded lead stats — mins: {LEAD_MINS}, maxs: {LEAD_MAXS}")

b_filt, a_filt = setup_filter(lowcut=0.5, highcut=40, fs=TARGET_HZ, order=3)

# ── Step 1: Load PTB-XL & find AFib records (proxy for Stroke) ──────────────

ptbxl_db = pd.read_csv(os.path.join(PTB_XL_PATH, "ptbxl_database.csv"))
scp_df   = pd.read_csv(os.path.join(PTB_XL_PATH, "scp_statements.csv"), index_col=0)

afib_codes = scp_df[
    scp_df.index.str.upper().str.contains('AFIB', na=False)
].index.tolist()

print(f"AFib SCP codes found: {afib_codes}")

def has_afib(scp_str):
    try:
        codes = eval(scp_str)
        return any(c in afib_codes for c in codes.keys())
    except Exception:
        return False

afib_records = ptbxl_db[ptbxl_db['scp_codes'].apply(has_afib)].reset_index(drop=True)
print(f"Total AFib records in PTB-XL: {len(afib_records)}")

afib_sample = afib_records.sample(n=N_SAMPLES, random_state=42).reset_index(drop=True)
print(f"\nSampled {N_SAMPLES} AFib records (proxy for stroke):")
print(afib_sample[['ecg_id', 'patient_id', 'scp_codes', 'age', 'sex']].to_string(index=False))

# ── Step 2: Load & preprocess ECG signals ────────────────────────────────────

def load_ptbxl_ecg(row, ptb_path):
    fname  = os.path.join(ptb_path, row['filename_lr'])
    record = wfdb.rdrecord(fname)
    return record.p_signal.astype(np.float32)

def preprocess_ecg(signal, lead_indices, target_length, lead_mins, lead_maxs,
                   src_hz=100, tgt_hz=128):
    sig = signal[:, lead_indices]

    if src_hz != tgt_hz:
        g   = gcd(src_hz, tgt_hz)
        up  = tgt_hz // g
        dn  = src_hz // g
        sig = resample_poly(sig, up, dn, axis=0).astype(np.float32)

    sig = bandpass_filter(sig, b_filt, a_filt).astype(np.float32)

    segment, found = extract_centered_segment_ptb(sig, fs=tgt_hz, segment_length=target_length)
    if not found:
        T = sig.shape[0]
        if T >= target_length:
            segment = sig[:target_length, :]
        else:
            pad = np.zeros((target_length - T, sig.shape[1]), dtype=np.float32)
            segment = np.concatenate([sig, pad], axis=0)

    segment = segment.astype(np.float32)
    scaled  = np.copy(segment)
    n_leads = segment.shape[1]
    for lead_idx in range(n_leads):
        denom = lead_maxs[lead_idx] - lead_mins[lead_idx]
        if denom == 0:
            denom = 1e-12
        scaled[:, lead_idx] = (segment[:, lead_idx] - lead_mins[lead_idx]) / denom
        scaled[:, lead_idx] = scaled[:, lead_idx] * 2 - 1

    return scaled

ecg_signals, meta_rows = [], []
for _, row in afib_sample.iterrows():
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
        f"  Example expected path: {os.path.join(PTB_XL_PATH, afib_sample.iloc[0]['filename_lr'])}"
    )

ecg_array  = np.stack(ecg_signals, axis=0)
ecg_tensor = torch.tensor(ecg_array, dtype=torch.float32).permute(0, 2, 1)
n_loaded   = ecg_tensor.shape[0]
print(f"Loaded {n_loaded} ECG signals — shape: {ecg_tensor.shape}")

# ── Step 3: Run through classifier (ECG only — no CRF input) ─────────────────

stroke_labels = torch.full((n_loaded,), STROKE_LABEL, dtype=torch.long)

model = Classifier(num_leads=N_LEADS, num_risk_factors=0, num_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load(CLASSIFIER_PTH, map_location=device))
model.eval()

dataset    = TensorDataset(ecg_tensor, stroke_labels)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

all_preds, all_probs = [], []
with torch.no_grad():
    for ecg_b, _ in dataloader:
        logits = model(ecg_b.to(device))
        probs  = torch.softmax(logits, dim=1)
        preds  = torch.argmax(probs, dim=1)
        all_preds.append(preds.cpu())
        all_probs.append(probs.cpu())

all_preds = torch.cat(all_preds).numpy()
all_probs = torch.cat(all_probs).numpy()

# ── Step 4: Results ───────────────────────────────────────────────────────────

meta_df = pd.DataFrame(meta_rows)[['ecg_id', 'patient_id', 'age', 'sex', 'scp_codes']].reset_index(drop=True)
results = meta_df.copy()
results['true_label']      = 'Stroke'
results['predicted_label'] = [CLASS_NAMES[p] for p in all_preds]
results['correct']         = results['predicted_label'] == 'Stroke'
results['prob_None']       = all_probs[:, 0]
results['prob_MI']         = all_probs[:, 1]
results['prob_Stroke']     = all_probs[:, 2]
results['confidence']      = all_probs.max(axis=1)

print("\n── Results (ECG-only model) ─────────────────────────────────────")
print(results[['ecg_id', 'age', 'sex', 'predicted_label',
               'correct', 'prob_Stroke', 'confidence']].to_string(index=False))

correct = results['correct'].sum()
print(f"\nAccuracy on PTB-XL AFib samples (ECG-only, stroke proxy): {correct}/{n_loaded} ({correct/n_loaded*100:.1f}%)")
print("\nPrediction breakdown:")
print(results['predicted_label'].value_counts().to_string())

results.to_csv("ptbxl_afib_stroke_proxy_ecg_only_results.csv", index=False)
print("\nSaved: ptbxl_afib_stroke_proxy_ecg_only_results.csv")