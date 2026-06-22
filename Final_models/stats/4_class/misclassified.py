import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from Final_models.WGAN.preprocessing_utils import reorder_features
from Final_models.Classifier.classifier import Classifier

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH  = "Final_models/Classifier/models/classifier18/best_model.pth"
BATCH_SIZE  = 64
CLASS_NAMES = {0: 'None', 1: 'MI', 2: 'Stroke', 3: 'Syncope'}
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load model ────────────────────────────────────────────────────────────────

model = Classifier(num_leads=3, num_risk_factors=7, num_classes=4).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ── Rebuild real test set (same as your CWGAN main() / training script) ──────

ecg_data = np.load("real_ecg.npy", allow_pickle=True)
crf_data = np.load("real_crf.npy", allow_pickle=True).tolist()

vasc_events  = [val['Vascular event'] for val in crf_data]
keys         = [k for k in crf_data[0].keys() if k != 'Vascular event']
non_vasc     = np.array([[d[k] for k in keys] for d in crf_data])
non_vasc_reordered = np.array([reorder_features(row) for row in non_vasc])

ecg_tensor   = torch.tensor(ecg_data, dtype=torch.float32).permute(0, 2, 1)
crf_tensor   = torch.tensor(non_vasc_reordered, dtype=torch.float32)
label_tensor = torch.tensor(vasc_events, dtype=torch.long)

dataset    = TensorDataset(ecg_tensor, crf_tensor, label_tensor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# ── Run inference ─────────────────────────────────────────────────────────────

all_preds, all_probs, all_labels = [], [], []

with torch.no_grad():
    for ecg_b, crf_b, label_b in dataloader:
        if label_b.dim() == 2 and label_b.size(1) == 1:
            label_b = label_b.squeeze(1)

        logits = model(ecg_b.to(device), crf_b.to(device))
        probs  = torch.softmax(logits, dim=1)
        preds  = torch.argmax(probs, dim=1)

        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_labels.append(label_b.cpu().numpy())

all_preds  = np.concatenate(all_preds)
all_probs  = np.concatenate(all_probs)
all_labels = np.concatenate(all_labels)

# ── Build full results table with CRF values ──────────────────────────────────

crf_df = pd.DataFrame(non_vasc_reordered, columns=keys)
results = crf_df.copy()
results.insert(0, 'sample_idx',      np.arange(len(all_labels)))
results.insert(1, 'true_label',      [CLASS_NAMES[i] for i in all_labels])
results.insert(2, 'predicted_label', [CLASS_NAMES[i] for i in all_preds])
results.insert(3, 'prob_None',       all_probs[:, 0])
results.insert(4, 'prob_MI',         all_probs[:, 1])
results.insert(5, 'prob_Stroke',     all_probs[:, 2])
results.insert(6, 'prob_Syncope',    all_probs[:, 3])

# ── Split into correct vs misclassified None patients ─────────────────────────

none_correct       = results[(results['true_label'] == 'None') & (results['predicted_label'] == 'None')]
none_as_mi         = results[(results['true_label'] == 'None') & (results['predicted_label'] == 'MI')]

print(f"None correctly classified: {len(none_correct)}")
print(f"None misclassified as MI: {len(none_as_mi)}")

# ── Compare CRF feature distributions between the two groups ──────────────────

crf_cols = keys  # the 7 risk factor column names

print("\n── CRF feature comparison: correct None vs None→MI ──────────────")
comparison = pd.DataFrame({
    'feature':           crf_cols,
    'correct_none_mean': [none_correct[c].mean() for c in crf_cols],
    'none_as_mi_mean':    [none_as_mi[c].mean() for c in crf_cols],
})
comparison['diff'] = comparison['none_as_mi_mean'] - comparison['correct_none_mean']
comparison = comparison.sort_values('diff', key=abs, ascending=False)
print(comparison.to_string(index=False))

# ── Confidence comparison — how confidently wrong are these predictions? ─────

print(f"\nAvg prob_MI for correctly-classified None: {none_correct['prob_MI'].mean():.4f}")
print(f"Avg prob_MI for None→MI misclassified:      {none_as_mi['prob_MI'].mean():.4f}")
print(f"Avg prob_None for None→MI misclassified:    {none_as_mi['prob_None'].mean():.4f}")

# ── Save for further inspection ────────────────────────────────────────────────

none_correct.to_csv('none_correct_samples.csv', index=False)
none_as_mi.to_csv('none_as_mi_samples.csv', index=False)
comparison.to_csv('none_vs_mi_crf_comparison.csv', index=False)

print("\nSaved:")
print("  none_correct_samples.csv       — the 109 correctly classified None patients")
print("  none_as_mi_samples.csv         — the 12 misclassified None\u2192MI patients")
print("  none_vs_mi_crf_comparison.csv  — feature-by-feature comparison between groups")