import torch
from ignite.metrics import MaximumMeanDiscrepancy as MMD
from torch.utils.data import TensorDataset, DataLoader
import os
from preprocessing_utils import per_lead_minmax_scaling
import numpy as np

BATCH_SIZE = 128

if os.path.exists("../../biased_ptbxl_ecgs.npy"):
    data = np.load("../../biased_ptbxl_ecgs.npy", allow_pickle=True)
    normalized_data, lead_mins, lead_maxs = per_lead_minmax_scaling(data)

normalized_data = np.array(normalized_data)
dataset_tensor = torch.tensor(normalized_data, dtype=torch.float32)

dataloader = DataLoader(
    TensorDataset(dataset_tensor),
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def flatten_ecg(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.size(0), -1)


num_epochs = 1

for epoch in range(num_epochs):
    mmd = MMD(var=1.0, device=device)
    mmd.reset()

    running_mmd = 0.0
    num_steps = 0

    for i, (real_ecg,) in enumerate(dataloader):
        with torch.no_grad():
            perm = torch.randperm(real_ecg.size(0))
            half = real_ecg.size(0) // 2

            real_shuffled = real_ecg[perm]
            real_1 = real_shuffled[:half]
            real_2 = real_shuffled[half:2 * half]

            x = flatten_ecg(real_1).to(device)
            y = flatten_ecg(real_2).to(device)

            mmd.reset()
            mmd.update((x, y))
            val = mmd.compute()

            print(f"Step: {i} | MMD: {val:.8f}")

            running_mmd += val
            num_steps += 1

    print(f"Avg MMD over epoch: {running_mmd / num_steps:.8f}")
