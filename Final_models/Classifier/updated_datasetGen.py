import os
import random
import numpy as np
import pandas as pd
import torch

from torch.utils.data import TensorDataset, Subset, DataLoader
from sklearn.model_selection import train_test_split

from sdv.single_table import TVAESynthesizer
from sdv.metadata import SingleTableMetadata

from Final_models.CWGAN.UpsampleAndConv1D import Generator
from Final_models.CWGAN.UpsampleAndConv1D import Critic


# -----------------------------
# ECG generation
# -----------------------------
@torch.no_grad()
def generate_ecgs_for_labels(
    label_tensor: torch.Tensor,
    generator: torch.nn.Module,
    noise_dim: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate ECGs conditioned on labels.

    Returns:
        (N, leads, timestep)
    """
    generator = generator.to(device).eval()
    num_samples = label_tensor.shape[0]
    out = []

    for i in range(0, num_samples, batch_size):
        y = label_tensor[i:i + batch_size].to(device)
        b = y.size(0)
        z = torch.randn(b, noise_dim, device=device)
        x = generator(z, y.unsqueeze(1))
        x = x.permute(0, 2, 1).contiguous()  # (B, leads, T)
        out.append(x.detach().cpu())

    return torch.cat(out, dim=0)


# -----------------------------
# Critic scoring
# -----------------------------
@torch.no_grad()
def score_wgan(
    critic: torch.nn.Module,
    x_ecg: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    critic = critic.to(device).eval()
    scores = []
    for i in range(0, x_ecg.size(0), batch_size):
        xb = x_ecg[i:i + batch_size].to(device)
        s = critic(xb)
        scores.append(s.view(-1).detach().cpu())
    return torch.cat(scores, dim=0)


@torch.no_grad()
def score_cwgan(
    critic: torch.nn.Module,
    x_ecg: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Score ECGs with your CWGAN critic.

    Includes a guard for batch_size==1 (FiLM + squeeze bugs).
    """
    critic = critic.to(device).eval()
    scores = []

    for i in range(0, x_ecg.size(0), batch_size):
        xb = x_ecg[i:i + batch_size]
        yb = y[i:i + batch_size]

        # Avoid batch==1 (common FiLM + squeeze issue)
        take_first_only = False
        if xb.size(0) == 1:
            xb = xb.repeat(2, 1, 1)
            yb = yb.repeat(2)
            take_first_only = True

        xb = xb.to(device)
        yb_in = yb.unsqueeze(1).to(device)  # your Critic expects (B,1)

        s = critic(xb, yb_in).view(-1).detach().cpu()
        if take_first_only:
            s = s[:1]
        scores.append(s)

    return torch.cat(scores, dim=0)


def keep_top_pct_per_class(scores: torch.Tensor, labels: torch.Tensor, pct: float) -> torch.Tensor:
    keep = []
    for c in labels.unique():
        idx = (labels == c).nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            continue
        sc = scores[idx]
        k = max(1, int(pct * idx.numel()))
        topk = torch.topk(sc, k=k, largest=True).indices
        keep.append(idx[topk])
    if len(keep) == 0:
        return torch.empty((0,), dtype=torch.long)
    return torch.cat(keep, dim=0)


@torch.no_grad()
def compute_real_threshold_cwgan(
    critic: torch.nn.Module,
    real_ecg: torch.Tensor,
    real_labels: torch.Tensor,
    device: torch.device,
    batch_size: int,
    quantile: float = 0.10,
) -> float:
    scores = score_cwgan(critic, real_ecg.float(),
                         real_labels.long(), batch_size, device)
    return float(torch.quantile(scores, quantile).item())


# -----------------------------
# TVAE CRF helpers
# -----------------------------
def load_tvae_model(
    df_csv_path: str,
    tvae_pkl_path: str,
    epochs: int = 3000,
    cuda: bool = True,
) -> TVAESynthesizer:
    df = pd.read_csv(df_csv_path)
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)

    synth = TVAESynthesizer(
        metadata=metadata, epochs=epochs, cuda=cuda, verbose=True)
    synth = synth.load(tvae_pkl_path)
    return synth


def sample_crfs(
    tvae: TVAESynthesizer,
    n: int,
    condition_mapping: dict,
) -> tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
    df = tvae.sample(n)
    labels = df['Vascular event'].astype(
        str).str.lower().map(condition_mapping)
    valid = labels.notna()
    df = df.loc[valid].reset_index(drop=True)
    labels = labels.loc[valid].astype(int)

    feats = df.drop(columns=['Vascular event'])

    y = torch.tensor(labels.to_numpy(), dtype=torch.long)
    x = torch.tensor(feats.to_numpy(), dtype=torch.float32)
    return y, x, df


def get_crf_feature_dim(tvae: TVAESynthesizer, condition_mapping: dict) -> int:
    _, x_tmp, _ = sample_crfs(tvae, n=8, condition_mapping=condition_mapping)
    if x_tmp.numel() == 0:
        raise RuntimeError(
            "TVAE returned no valid rows when inferring CRF dim.")
    return int(x_tmp.size(1))


# -----------------------------
# Generate + filter until target is reached
# -----------------------------
def generate_filtered_synthetic(
    tvae: TVAESynthesizer,
    generator: torch.nn.Module,
    cwgan_critic: torch.nn.Module,
    device: torch.device,
    condition_mapping: dict,
    target_total: int,
    per_class_targets: dict | None,
    noise_dim: int,
    batch_size: int,
    crf_feature_dim: int,
    chunk_candidates: int = 2048,
    keep_mode: str = "threshold",   # "threshold" or "top_pct"
    real_threshold: float | None = None,
    top_pct: float = 0.6,
    # completion helpers:
    max_loops: int = 300,
    fallback_to_top_pct: bool = True,
    fallback_start_loop: int = 150,
    fallback_top_pct: float = 0.30,
    adaptive_relax_threshold: bool = True,
    relax_every: int = 50,
    relax_multiplier: float = 0.25,   # lower by 0.25 * std each relax
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    kept_ecg: list[torch.Tensor] = []
    kept_crf: list[torch.Tensor] = []
    kept_y: list[torch.Tensor] = []

    if per_class_targets is not None:
        counts = {c: 0 for c in per_class_targets.keys()}
    else:
        counts = None

    loops = 0
    print_every = 10
    total_seen = 0
    total_accepted = 0

    # keep a mutable threshold we can relax
    thr = real_threshold

    while True:
        loops += 1
        if loops > max_loops:
            print("\n[Warning] Max loops reached.")
            print(f"  Accepted so far: {total_accepted}")
            if per_class_targets is not None:
                remaining_by_class = {
                    c: (per_class_targets[c] - counts[c]) for c in per_class_targets}
                print(f"  Remaining by class: {remaining_by_class}")
            print("  Returning what we have (increase max_loops / chunk_candidates, relax threshold, or use top_pct).\n")
            break

        # How many do we still need?
        if per_class_targets is None:
            need = target_total - sum(x.size(0) for x in kept_y)
            if need <= 0:
                break
            n_candidates = min(chunk_candidates, max(need * 2, batch_size))
        else:
            remaining = sum(
                max(0, per_class_targets[c] - counts[c]) for c in per_class_targets)
            if remaining <= 0:
                break
            n_candidates = min(chunk_candidates, max(
                remaining * 2, batch_size))

        total_seen += int(n_candidates)

        # 1) sample CRFs + labels
        y_cand, crf_cand, _ = sample_crfs(
            tvae, n_candidates, condition_mapping)
        if y_cand.numel() == 0:
            continue

        # only keep labels for classes that still need filling
        if per_class_targets is not None:
            mask_need = torch.zeros_like(y_cand, dtype=torch.bool)
            for c in per_class_targets.keys():
                if counts[c] < per_class_targets[c]:
                    mask_need |= (y_cand == c)
            y_cand = y_cand[mask_need]
            crf_cand = crf_cand[mask_need]
            if y_cand.numel() == 0:
                continue

        # 2) generate ECGs
        ecg_cand = generate_ecgs_for_labels(
            y_cand, generator, noise_dim, batch_size, device)

        # 3) score
        scores = score_cwgan(cwgan_critic, ecg_cand.float(),
                             y_cand.long(), batch_size, device)

        # 4) pick keep indices
        if keep_mode == "threshold":
            if thr is None:
                raise ValueError(
                    "keep_mode='threshold' requires real_threshold")

            keep_mask = scores >= thr
            keep_idx = keep_mask.nonzero(as_tuple=False).view(-1)

            # If stuck: relax threshold every so often
            if adaptive_relax_threshold and (loops % relax_every == 0) and keep_idx.numel() == 0:
                s_std = float(scores.std().item() + 1e-8)
                thr = thr - relax_multiplier * s_std
                keep_mask = scores >= thr
                keep_idx = keep_mask.nonzero(as_tuple=False).view(-1)
                print(
                    f"[Relax] loop={loops} new_thr={thr:.6f} kept={keep_idx.numel()}")

            # If still stuck: fallback to top_pct
            if fallback_to_top_pct and (loops >= fallback_start_loop) and keep_idx.numel() == 0:
                keep_idx = keep_top_pct_per_class(
                    scores, y_cand.long(), pct=fallback_top_pct)
                print(
                    f"[Fallback] loop={loops} top_pct={fallback_top_pct} kept={keep_idx.numel()}")

        elif keep_mode == "top_pct":
            keep_idx = keep_top_pct_per_class(
                scores, y_cand.long(), pct=top_pct)
        else:
            raise ValueError("keep_mode must be 'threshold' or 'top_pct'")

        kept_pre_cap = int(keep_idx.numel())
        if kept_pre_cap == 0:
            if loops % print_every == 0:
                if per_class_targets is None:
                    need = target_total - sum(x.size(0) for x in kept_y)
                    print(
                        f"[Loop {loops}] cand={n_candidates} kept=0 | accepted={total_accepted} | remaining={need}")
                else:
                    remaining_by_class = {
                        c: (per_class_targets[c] - counts[c]) for c in per_class_targets}
                    print(
                        f"[Loop {loops}] cand={n_candidates} kept=0 | counts={counts} | remaining={remaining_by_class}")
            continue

        ecg_keep = ecg_cand[keep_idx]
        crf_keep = crf_cand[keep_idx]
        y_keep = y_cand[keep_idx]

        # 5) apply per-class caps
        if per_class_targets is not None:
            final_mask = torch.zeros_like(y_keep, dtype=torch.bool)
            for c in per_class_targets.keys():
                idx_c = (y_keep == c).nonzero(as_tuple=False).view(-1)
                if idx_c.numel() == 0:
                    continue
                space = per_class_targets[c] - counts[c]
                if space <= 0:
                    continue
                take = min(space, idx_c.numel())
                final_mask[idx_c[:take]] = True
                counts[c] += take

            y_keep = y_keep[final_mask]
            ecg_keep = ecg_keep[final_mask]
            crf_keep = crf_keep[final_mask]

        kept_post_cap = int(y_keep.numel())
        total_accepted += kept_post_cap

        if (loops % print_every) == 0 or kept_post_cap > 0:
            if per_class_targets is None:
                need = target_total - sum(x.size(0) for x in kept_y)
                print(
                    f"[Loop {loops}] cand={n_candidates} kept_pre_cap={kept_pre_cap} kept_post_cap={kept_post_cap} "
                    f"seen={total_seen} accepted={total_accepted} remaining={need} thr={thr if thr is not None else 'NA'}"
                )
            else:
                remaining_by_class = {
                    c: (per_class_targets[c] - counts[c]) for c in per_class_targets}
                print(
                    f"[Loop {loops}] cand={n_candidates} kept_pre_cap={kept_pre_cap} kept_post_cap={kept_post_cap} "
                    f"counts={counts} remaining={remaining_by_class} thr={thr if thr is not None else 'NA'}"
                )

        if kept_post_cap > 0:
            kept_ecg.append(ecg_keep)
            kept_crf.append(crf_keep)
            kept_y.append(y_keep)

    # Build outputs (always with correct feature dimension)
    ecg = torch.cat(kept_ecg, dim=0) if len(
        kept_ecg) else torch.empty((0, 3, 640), dtype=torch.float32)
    crf = torch.cat(kept_crf, dim=0) if len(kept_crf) else torch.empty(
        (0, crf_feature_dim), dtype=torch.float32)
    y = torch.cat(kept_y, dim=0) if len(
        kept_y) else torch.empty((0,), dtype=torch.long)

    return ecg, crf, y


def main():
    # ------------------
    # Settings
    # ------------------
    real_data_fraction = 0.5
    number_of_samples = 10000
    BATCH_SIZE = 128
    noise_dim = 100

    # Generation/filtering
    chunk_candidates = 12000        # bigger chunks -> easier to fill quotas
    keep_mode = "threshold"        # "threshold" or "top_pct"
    real_quantile = 0.05           # lower -> easier pass (e.g., 0.05)
    top_pct = 1.0               # only used if keep_mode == "top_pct"

    # Completion knobs
    max_loops = 2000

    # Paths
    augmented_csv = "augmented_dataset.csv"
    fine_tune_npy = "fine_tune_data.npy"
    tvae_pkl = "TVAE_model.pkl"

    cgan_ckpt = "Final_models/CWGAN/models/UpsampleAndCNN_CWGAN/Model_2_GP_10.0_DTW_1.0/Model.pth"

    out_dir = "Final_models/Classifier/synth_datasets"
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    condition_mapping = {
        'none': 0,
        'myocardial infarction': 1,
        'stroke': 2,
        'syncope': 3
    }

    # ------------------
    # Load real data
    # ------------------
    real_data = np.load(fine_tune_npy, allow_pickle=True)
    real_ecg_list, real_labels_list = [], []
    for item in real_data:
        real_ecg_list.append(item[0])
        real_labels_list.append(item[1])

    real_shuffle = list(zip(real_ecg_list, real_labels_list))
    random.shuffle(real_shuffle)

    samples = real_shuffle[:number_of_samples]
    real_samples = samples[:int(len(samples) * real_data_fraction)]

    if len(real_samples) > 0:
        real_ecg_list_split, real_labels_split = zip(*real_samples)
        real_ecg_array = np.stack(real_ecg_list_split, axis=0)
        real_ecg_array = np.transpose(
            real_ecg_array, (0, 2, 1))  # (N, leads, T)
        real_ecg_tensor = torch.tensor(real_ecg_array, dtype=torch.float32)
        real_labels_tensor = torch.tensor(real_labels_split, dtype=torch.long)
    else:
        real_ecg_tensor = None
        real_labels_tensor = None

    # ------------------
    # Load TVAE
    # ------------------
    tvae = load_tvae_model(augmented_csv, tvae_pkl,
                           epochs=3000, cuda=torch.cuda.is_available())
    crf_feature_dim = get_crf_feature_dim(tvae, condition_mapping)

    # ------------------
    # Load generator + critic
    # ------------------
    generator = Generator(ecg_length=640, n_leads=3, latent_dim=noise_dim)
    cgan_model = torch.load(cgan_ckpt, map_location=device, weights_only=False)
    generator.load_state_dict(cgan_model['gen_state_dict'])

    cwgan_critic = Critic()
    cwgan_critic.load_state_dict(cgan_model['critic_state_dict'])

    # ------------------
    # How many synthetic needed?
    # ------------------
    if real_ecg_tensor is not None:
        synth_needed = number_of_samples - real_ecg_tensor.size(0)
    else:
        synth_needed = number_of_samples

    per_class = synth_needed // 4
    per_class_targets = {
        0: per_class,
        1: per_class,
        2: per_class,
        3: synth_needed - 3 * per_class
    }

    # ------------------
    # Threshold calibration
    # ------------------
    real_threshold = None
    if keep_mode == "threshold":
        if real_ecg_tensor is not None:
            real_threshold = compute_real_threshold_cwgan(
                cwgan_critic,
                real_ecg_tensor,
                real_labels_tensor,
                device=device,
                batch_size=BATCH_SIZE,
                quantile=real_quantile
            )
        else:
            real_threshold = 2.0

        print(
            f"[Info] CWGAN threshold from real q={real_quantile}: {real_threshold:.6f}")

    # ------------------
    # Generate synthetic
    # ------------------
    if synth_needed > 0:
        synthetic_ecg_tensor, synthetic_crf_tensor, synthetic_labels_tensor = generate_filtered_synthetic(
            tvae=tvae,
            generator=generator,
            cwgan_critic=cwgan_critic,
            device=device,
            condition_mapping=condition_mapping,
            target_total=synth_needed,
            per_class_targets=per_class_targets,
            noise_dim=noise_dim,
            batch_size=BATCH_SIZE,
            crf_feature_dim=crf_feature_dim,
            chunk_candidates=chunk_candidates,
            keep_mode=keep_mode,
            real_threshold=real_threshold,
            top_pct=top_pct,
            max_loops=max_loops,
            fallback_to_top_pct=False,
            fallback_start_loop=1000,
            fallback_top_pct=0.30,
            adaptive_relax_threshold=True,
            relax_every=10,
            relax_multiplier=0.5,
        )

        # If we still came up short, pad by switching to top_pct mode to fill the rest
        if synthetic_labels_tensor.numel() < synth_needed:
            short = synth_needed - synthetic_labels_tensor.numel()
            print(
                f"[Warning] Only generated {synthetic_labels_tensor.numel()} / {synth_needed}. Filling remaining {short} with top_pct mode.")

            # fill remainder without strict per-class quotas
            fill_ecg, fill_crf, fill_y = generate_filtered_synthetic(
                tvae=tvae,
                generator=generator,
                cwgan_critic=cwgan_critic,
                device=device,
                condition_mapping=condition_mapping,
                target_total=short,
                per_class_targets=None,
                noise_dim=noise_dim,
                batch_size=BATCH_SIZE,
                crf_feature_dim=crf_feature_dim,
                chunk_candidates=chunk_candidates,
                keep_mode="top_pct",
                real_threshold=None,
                top_pct=0.5,
                max_loops=max_loops,
                fallback_to_top_pct=False,
                adaptive_relax_threshold=False,
            )

            synthetic_ecg_tensor = torch.cat(
                [synthetic_ecg_tensor, fill_ecg], dim=0)
            synthetic_crf_tensor = torch.cat(
                [synthetic_crf_tensor, fill_crf], dim=0)
            synthetic_labels_tensor = torch.cat(
                [synthetic_labels_tensor, fill_y], dim=0)

            # trim in case we overshoot
            synthetic_ecg_tensor = synthetic_ecg_tensor[:synth_needed]
            synthetic_crf_tensor = synthetic_crf_tensor[:synth_needed]
            synthetic_labels_tensor = synthetic_labels_tensor[:synth_needed]

    else:
        synthetic_ecg_tensor = torch.empty((0, 3, 640), dtype=torch.float32)
        synthetic_crf_tensor = torch.empty(
            (0, crf_feature_dim), dtype=torch.float32)
        synthetic_labels_tensor = torch.empty((0,), dtype=torch.long)

    # ------------------
    # Build real CRFs (matched by label)
    # ------------------
    if real_ecg_tensor is not None:
        y_real = real_labels_tensor
        y_tvae, x_tvae, _ = sample_crfs(tvae, n=max(
            32, len(y_real) * 2), condition_mapping=condition_mapping)

        groups = {code: [] for code in condition_mapping.values()}
        for i in range(y_tvae.numel()):
            groups[int(y_tvae[i].item())].append(x_tvae[i])

        matched_real_crfs = []
        for lab in y_real.tolist():
            bucket = groups.get(int(lab), [])
            if len(bucket) > 0:
                matched_real_crfs.append(
                    bucket[random.randrange(len(bucket))].unsqueeze(0))
            else:
                matched_real_crfs.append(torch.zeros(
                    (1, crf_feature_dim), dtype=torch.float32))

        real_crf_tensor = torch.cat(matched_real_crfs, dim=0)

        combined_ecg = torch.cat(
            [real_ecg_tensor, synthetic_ecg_tensor], dim=0)
        combined_crf = torch.cat(
            [real_crf_tensor, synthetic_crf_tensor], dim=0)
        combined_y = torch.cat(
            [real_labels_tensor, synthetic_labels_tensor], dim=0)

        combined_dataset = TensorDataset(
            combined_ecg, combined_crf, combined_y)
    else:
        combined_dataset = TensorDataset(
            synthetic_ecg_tensor, synthetic_crf_tensor, synthetic_labels_tensor)

    # ------------------
    # Split train/valid/test
    # ------------------
    indices = np.arange(len(combined_dataset))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=42)

    train_dataset = Subset(combined_dataset, train_idx)
    val_dataset = Subset(combined_dataset, val_idx)
    test_dataset = Subset(combined_dataset, test_idx)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    datasets = {"train": train_loader,
                "valid": valid_loader, "test": test_loader}

    out_path = os.path.join(
        out_dir, f"{real_data_fraction}_real_synth_dataset_filtered.pth")
    torch.save(datasets, out_path)
    print(f"[Done] Saved filtered dataset loaders -> {out_path}")


if __name__ == "__main__":
    main()
