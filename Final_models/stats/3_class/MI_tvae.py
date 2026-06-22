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
from sdv.single_table import TVAESynthesizer
from sdv.metadata import SingleTableMetadata

# ── Config ────────────────────────────────────────────────────────────────────

PTB_XL_PATH      = "data/pretrain"
CLASSIFIER_PTH   = "Final_models/Classifier/models/classifier34/best_model.pth"   # update N to match your training run
LEAD_STATS_PATH  = "lead_minmax_stats.npz"
TVAE_PKL         = "TVAE_model.pkl"
AUGMENTED_CSV    = "augmented_dataset.csv"

N_SAMPLES      = 10
BATCH_SIZE     = 64
NOISE_DIM      = 100
ECG_LENGTH     = 640
N_LEADS        = 3
SAMPLING_RATE  = 100
TARGET_HZ      = 128
LEAD_INDICES   = [2, 8, 10]   # III, V3, V5

CLASS_NAMES    = {0: 'None', 1: 'MI', 2: 'Stroke'}   # 3-class — Syncope dropped
NUM_CLASSES    = 3
# NOTE: CONDITION_MAP still includes syncope since the TVAE was trained on the
# original 4-class augmented_dataset.csv — we filter it out after sampling.
CONDITION_MAP  = {'none': 0, 'myocardial infarction': 1, 'stroke': 2, 'syncope': 3}
MI_LABEL       = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load training-derived lead min/max stats ──────────────────────────────────

stats     = np.load(LEAD_STATS_PATH)
LEAD_MINS = stats['lead_mins']
LEAD_MAXS = stats['lead_maxs']
print(f"Loaded lead stats — mins: {LEAD_MINS}, maxs: {LEAD_MAXS}")

b_filt, a_filt = setup_filter(lowcut=0.5, highcut=40, fs=TARGET_HZ, order=3)

# ── Step 1: Load PTB-XL database & find MI records ───────────────────────────

ptbxl_db = pd.read_csv(os.path.join(PTB_XL_PATH, "ptbxl_database.csv"))
scp_df   = pd.read_csv(os.path.join(PTB_XL_PATH, "scp_statements.csv"), index_col=0)

mi_codes = scp_df[
    scp_df['diagnostic_class'].str.upper().str.contains('MI', na=False) |
    scp_df.index.str.upper().str.contains('MI', na=False)
].index.tolist()

print(f"MI SCP codes found: {mi_codes}")

def has_mi(scp_str):
    try:
        codes = eval(scp_str)
        return any(c in mi_codes for c in codes.keys())
    except Exception:
        return False

mi_records = ptbxl_db[ptbxl_db['scp_codes'].apply(has_mi)].reset_index(drop=True)
print(f"Total MI records in PTB-XL: {len(mi_records)}")

mi_sample = mi_records.sample(n=N_SAMPLES, random_state=42).reset_index(drop=True)
print(f"\nSampled {N_SAMPLES} MI records:")
print(mi_sample[['ecg_id', 'patient_id', 'scp_codes', 'age', 'sex']].to_string(index=False))

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
        "No ECG signals were loaded. Check PTB_XL_PATH is correct.\n"
        f"  PTB_XL_PATH = {PTB_XL_PATH}\n"
        f"  Example expected path: {os.path.join(PTB_XL_PATH, mi_sample.iloc[0]['filename_lr'])}"
    )

ecg_array  = np.stack(ecg_signals, axis=0)
ecg_tensor = torch.tensor(ecg_array, dtype=torch.float32).permute(0, 2, 1)
print(f"\nLoaded ECG tensor shape: {ecg_tensor.shape}")

# ── Step 3: Generate matched CRF samples via TVAE ────────────────────────────

def sample_crfs(tvae, n, condition_mapping):
    df     = tvae.sample(n)
    labels = df['Vascular event'].astype(str).str.lower().map(condition_mapping)
    valid  = labels.notna()
    df     = df.loc[valid].reset_index(drop=True)
    labels = labels.loc[valid].astype(int)
    feats  = df.drop(columns=['Vascular event'])
    y      = torch.tensor(labels.to_numpy(), dtype=torch.long)
    x      = torch.tensor(feats.to_numpy(), dtype=torch.float32)
    return y, x, df

df       = pd.read_csv(AUGMENTED_CSV)
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df)
tvae     = TVAESynthesizer(metadata=metadata, epochs=3000,
                           cuda=torch.cuda.is_available(), verbose=False)
tvae     = tvae.load(TVAE_PKL)

mi_crfs  = []
attempts = 0
while len(mi_crfs) < N_SAMPLES and attempts < 50:
    y_cand, x_cand, _ = sample_crfs(tvae, n=N_SAMPLES * 10,
                                     condition_mapping=CONDITION_MAP)
    mi_mask = (y_cand == MI_LABEL)
    mi_x    = x_cand[mi_mask]
    mi_crfs.append(mi_x)
    attempts += 1

mi_crf_tensor = torch.cat(mi_crfs, dim=0)[:N_SAMPLES]
n_loaded      = ecg_tensor.shape[0]
mi_labels     = torch.full((n_loaded,), MI_LABEL, dtype=torch.long)

print(f"Generated CRF tensor shape: {mi_crf_tensor.shape}")
print(f"CRF feature dim: {mi_crf_tensor.shape[1]}")

# ── Step 4: Run through classifier ───────────────────────────────────────────

model = Classifier(num_leads=N_LEADS,
                   num_risk_factors=mi_crf_tensor.shape[1],
                   num_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load(CLASSIFIER_PTH, map_location=device))
model.eval()

dataset    = TensorDataset(ecg_tensor, mi_crf_tensor, mi_labels)
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
results['confidence']      = all_probs.max(axis=1)

print("\n── Results ──────────────────────────────────────────────────────")
print(results[['ecg_id', 'age', 'sex', 'predicted_label',
               'correct', 'prob_MI', 'confidence']].to_string(index=False))

correct = results['correct'].sum()
print(f"\nAccuracy on PTB-XL MI samples: {correct}/{n_loaded} ({correct/n_loaded*100:.1f}%)")
print("\nPrediction breakdown:")
print(results['predicted_label'].value_counts().to_string())

results.to_csv("ptbxl_mi_tvae_3class_results.csv", index=False)
print("\nSaved: ptbxl_mi_tvae_3class_results.csv")