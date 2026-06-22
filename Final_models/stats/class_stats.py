import torch

datasets = torch.load("synth_datasets/0.5_real_synth_dataset_filtered.pth", weights_only=False)

for split_name, loader in datasets.items():
    all_labels = []
    for ecg, crf, label in loader:
        all_labels.extend(label.tolist())
    
    import numpy as np
    unique, counts = np.unique(all_labels, return_counts=True)
    CLASS_NAMES = {0: 'None', 1: 'MI', 2: 'Stroke', 3: 'Syncope'}
    print(f"\n{split_name} split — total: {len(all_labels)}")
    for label, count in zip(unique, counts):
        pct = count / len(all_labels) * 100
        print(f"  {CLASS_NAMES.get(int(label), label):10s}: {count:5d} ({pct:.1f}%)")