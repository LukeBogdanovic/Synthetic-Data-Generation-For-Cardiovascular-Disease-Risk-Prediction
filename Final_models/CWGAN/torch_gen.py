import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import streamlit as st

plt.rcParams.update({"font.size":16})

from pathlib import Path
import sys

WGAN_PARENT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WGAN_PARENT))
from Final_models.WGAN.preprocessing_utils import per_lead_minmax_scaling, per_lead_inverse_scaling, bandpass_filter, setup_filter
from Final_models.CWGAN.UpsampleAndConv1D import Generator as UPConv
from Final_models.CWGAN.BiLSTM_CNN import Generator as BiLSTM
from Final_models.CWGAN.ConvTranspose1D import Generator as ConvT


latent_dim: int = 100
ecg_length: int = 128 * 5
n_leads: int = 3

GEN_UPCONV_CKPT = "Final_models/CWGAN/models/UpsampleAndCNN_CWGAN/Model_1_GP_10.0_DTW_0.0/Model.pth"
GEN_BILSTM_CKPT = "Final_models/CWGAN/models/BiLSTM_CNN_CWGAN/Model_1_GP_10.0_DTW_0.0/Model.pth"
GEN_CONVT_CKPT = "Final_models/CWGAN/models/DCNN_WGAN/Model_1_GP_10.0_DTW_0.0/Model.pth"
GEN_UPCONV_DTW_CKPT = "Final_models/CWGAN/models/UpsampleAndCNN_CWGAN/Model_1_GP_10.0_DTW_1.0/Model.pth"
GEN_BILSTM_DTW_CKPT = "Final_models/CWGAN/models/BiLSTM_CNN_CWGAN/Model_1_GP_10.0_DTW_1.0/Model.pth"
GEN_CONVT_DTW_CKPT = "Final_models/CWGAN/models/DCNN_WGAN/Model_1_GP_10.0_DTW_1.0/Model.pth"


@st.cache_resource(show_spinner=True)
def setup_bp_filter() -> Tuple[np.ndarray, np.ndarray]:
    b, a = setup_filter()
    return b, a


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_lead_minmax(path: str) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing dataset file: {path}")
    data = np.load(path, allow_pickle=True)
    ecg_dataset = np.stack([item for item in data])
    _, lead_mins, lead_maxs = per_lead_minmax_scaling(
        ecg_dataset=ecg_dataset)
    return lead_mins, lead_maxs


@st.cache_resource(show_spinner=True)
def load_models_and_stats() -> Tuple[Dict[str, torch.nn.Module], np.ndarray, np.ndarray, torch.device]:
    device = get_device()
    lead_mins, lead_maxs = load_lead_minmax("biased_ptbxl_ecgs.npy")
    models: Dict[str, torch.nn.Module] = {}

    gen_up = UPConv(ecg_length=ecg_length, n_leads=n_leads,
                    latent_dim=latent_dim).to(device)
    ckpt_up = torch.load(
        GEN_UPCONV_CKPT, map_location=device, weights_only=False)
    gen_up.load_state_dict(ckpt_up['gen_state_dict'])
    gen_up.eval()
    models['upconv'] = gen_up

    gen_bi = BiLSTM().to(device)
    ckpt_bi = torch.load(
        GEN_BILSTM_CKPT, map_location=device, weights_only=False)
    gen_bi.load_state_dict(ckpt_bi['gen_state_dict'])
    gen_bi.eval()
    models['bilstm'] = gen_bi

    gen_ct = ConvT().to(device)
    ckpt_ct = torch.load(
        GEN_CONVT_CKPT, map_location=device, weights_only=False)
    gen_ct.load_state_dict(ckpt_ct['gen_state_dict'])
    gen_ct.eval()
    models['dcnn'] = gen_ct

    gen_up_dtw = UPConv(ecg_length=ecg_length, n_leads=n_leads,
                    latent_dim=latent_dim).to(device)
    ckpt_up_dtw = torch.load(
        GEN_UPCONV_DTW_CKPT, map_location=device, weights_only=False)
    gen_up_dtw.load_state_dict(ckpt_up_dtw['gen_state_dict'])
    gen_up_dtw.eval()
    models['upconv_dtw'] = gen_up_dtw

    gen_bi_dtw = BiLSTM().to(device)
    ckpt_bi_dtw = torch.load(
        GEN_BILSTM_DTW_CKPT, map_location=device, weights_only=False)
    gen_bi_dtw.load_state_dict(ckpt_bi_dtw['gen_state_dict'])
    gen_bi_dtw.eval()
    models['bilstm_dtw'] = gen_bi_dtw

    gen_ct_dtw = ConvT().to(device)
    ckpt_ct_dtw = torch.load(
        GEN_CONVT_DTW_CKPT, map_location=device, weights_only=False)
    gen_ct_dtw.load_state_dict(ckpt_ct_dtw['gen_state_dict'])
    gen_ct_dtw.eval()
    models['dcnn_dtw'] = gen_ct_dtw

    return models, lead_mins, lead_maxs, device


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    seed_int = int(seed)
    torch.manual_seed(seed_int)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_int)


@torch.no_grad()
def generate_sample(generator: torch.nn.Module, ecg_type: int, device: torch.device, seed: Optional[int]) -> np.ndarray:
    set_seed(seed)
    label = torch.tensor([ecg_type], dtype=torch.long, device=device).unsqueeze(1)
    out: torch.Tensor = generator(noise, label)
    return out.detach().cpu().numpy()


def standardize_sample_layout(sample: np.ndarray, model_name: str) -> np.ndarray:
    """Ensure sample is (B, T, leads) for downstream inverse-scaling/plotting."""
    if model_name == "bilstm":
        sample = np.transpose(sample, (0, 2, 1))

    if sample.ndim == 3 and sample.shape[-1] != n_leads and sample.shape[1] == n_leads:
        sample = np.transpose(sample, (0, 2, 1))

    return sample


def make_plot(sample: np.ndarray, lead_mins, lead_maxs, model_name: str, display_len: int,b,a) -> Figure:
    sample = per_lead_inverse_scaling(sample, lead_mins, lead_maxs)
    sample = sample.squeeze(0)
    sample = bandpass_filter(sample, b, a)
    T = sample.shape[0]
    display_len = int(max(128, min(T, display_len)))

    fig, axs = plt.subplots(n_leads, 1, figsize=(10, 8), sharex=True, dpi=150)

    lead_labels = ["Lead III", "Lead V3", "Lead V5"]
    x = np.arange(display_len)

    for i, ax in enumerate(axs):
        ax.plot(x, sample[:display_len, i], linewidth=2.0)
        ax.set_title(lead_labels[i])
        ax.set_xlabel("Time (samples)")
        ax.set_ylabel("Amplitude (mV)")
        ax.grid(True, alpha=0.5)
    fig.tight_layout()
    return fig


st.set_page_config(page_title="ECG Generator", layout="wide")
st.title("ECG Generator")

models, lead_mins, lead_maxs, device = load_models_and_stats()
ecg_classes:dict[str,int] = {'Normal Sinus Rhythm':0,'Myocardial Infarction':1,'Stroke':2,'Syncope':3}
b, a = setup_bp_filter()

if "gen_counter" not in st.session_state:
    st.session_state.gen_counter = 0
if "last_samples" not in st.session_state:
    st.session_state.last_samples = []  # type: ignore[assignment]
if "last_settings" not in st.session_state:
    st.session_state.last_settings = None  # type: ignore[assignment]



with st.sidebar:
    st.header("Controls")
    with st.form("controls", clear_on_submit=False):
        model_name = st.selectbox("Model", options=list(models.keys()))
        display_len = st.slider("Display length (samples)",
                                min_value=128, max_value=640, value=int(128*2.5), step=1)
        seed = st.number_input("Seed (Optional)", value=0, step=1)
        use_seed = st.checkbox("Use seed", value=False)

        ecg_label = st.selectbox("ECG Class", options=list(ecg_classes.keys()))

        num_samples = st.number_input(
            "Generate N", min_value=1, max_value=16, value=1, step=1)

        submitted = st.form_submit_button("Generate")

seed_val: Optional[int] = int(seed) if use_seed else None
noise = torch.randn(1, latent_dim, device=device)

if submitted:
    st.session_state.gen_counter += 1

    gen = models[model_name]

    ecg_type: int = ecg_classes[ecg_label]

    samples: list[np.ndarray] = []
    for _ in range(int(num_samples)):
        s = generate_sample(gen, ecg_type, device, seed_val)
        s = standardize_sample_layout(s, model_name)
        samples.append(s)

    st.session_state.last_samples = samples
    st.session_state.last_settings = {
        "model_name": model_name,
        "display_len": int(display_len),
        "seed": seed_val,
        "ecg_type": ecg_type,
        "ecg_label": ecg_label,
        "num_samples": int(num_samples),
    }

if st.session_state.last_settings is not None and len(st.session_state.last_samples) > 0:
    settings = st.session_state.last_settings

    st.subheader(
        f"Generated ECGs (run #{st.session_state.gen_counter}) | "
        f"Model: {settings['model_name']} | ECG Class: {settings['ecg_label']} | Display: {settings['display_len']} Samples"
    )

    cols = st.columns(min(int(settings["num_samples"]), 4))
    for idx, sample in enumerate(st.session_state.last_samples):
        fig = make_plot(
            sample,
            lead_mins,
            lead_maxs,
            settings["model_name"],
            int(settings["display_len"]),
            b,a
        )
        cols[idx % len(cols)].pyplot(fig, clear_figure=True)
else:
    st.info("Click **Generate** to create an ECG.")

