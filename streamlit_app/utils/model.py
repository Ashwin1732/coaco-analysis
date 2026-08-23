import json
from pathlib import Path

import joblib
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[2]
CLASS_CKPT = ROOT / "thermal_classifier" / "outputs" / "best_efficientnet_b0.pt"
ACOUSTIC_CKPT = ROOT / "acoustic_classifier" / "outputs" / "best_efficientnet_b0_acoustic.pt"

# Define default paths in case we run from a different root
if not CLASS_CKPT.exists():
    # Fallback to checking from current working dir
    CLASS_CKPT = Path("thermal_classifier/outputs/best_efficientnet_b0.pt")
    ACOUSTIC_CKPT = Path("acoustic_classifier/outputs/best_efficientnet_b0_acoustic.pt")

CLASSES = ["OR", "R", "UR"]
CLASS_LABELS = {
    "OR": "Overripe",
    "R": "Ripe",
    "UR": "Unripe",
}

THERMAL_WEIGHT = 0.55
ACOUSTIC_WEIGHT = 0.45

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_classifier(num_classes: int) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

@st.cache_resource
def load_models():
    """Loads and caches the PyTorch models."""
    if not CLASS_CKPT.exists():
        st.warning(f"Missing classifier checkpoint: {CLASS_CKPT}")
        return None

    ckpt = torch.load(CLASS_CKPT, map_location=device, weights_only=False)
    classes = ckpt.get("classes", CLASSES)
    clf = build_classifier(len(classes))
    clf.load_state_dict(ckpt["model_state"])
    clf.to(device).eval()

    acoustic_clf = None
    acoustic_classes: list[str] = []
    if ACOUSTIC_CKPT.exists():
        ackpt = torch.load(ACOUSTIC_CKPT, map_location=device, weights_only=False)
        acoustic_classes = ackpt.get("classes", ["R", "UR"])
        acoustic_clf = build_classifier(len(acoustic_classes))
        acoustic_clf.load_state_dict(ackpt["model_state"])
        acoustic_clf.to(device).eval()

    return {
        "classes": classes,
        "classifier": clf,
        "acoustic_classes": acoustic_classes,
        "acoustic_classifier": acoustic_clf,
    }

TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

FS = 12800

def hz_to_mel(f: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + f / 700.0)

def mel_to_hz(m: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

def mel_filterbank(n_fft: int, n_mels: int, fs: int) -> np.ndarray:
    f_max = fs / 2.0
    mels = np.linspace(hz_to_mel(np.array([0.0]))[0], hz_to_mel(np.array([f_max]))[0], n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / fs).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        if center == left: center += 1
        if right == center: right += 1
        for j in range(left, center):
            if 0 <= j < fb.shape[1]: fb[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if 0 <= j < fb.shape[1]: fb[i, j] = (right - j) / (right - center)
    return fb

def mel_spectrogram(x: np.ndarray, fs: int, n_fft: int = 1024, hop: int = 256, n_mels: int = 128) -> np.ndarray:
    window = np.hanning(n_fft).astype(np.float32)
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    n_frames = 1 + (len(x) - n_fft) // hop
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(n_frames, n_fft), strides=(x.strides[0] * hop, x.strides[0]), writeable=False
    ).copy()
    frames *= window
    spec = np.fft.rfft(frames, axis=1)
    power = (spec.real**2 + spec.imag**2).astype(np.float32)
    fb = mel_filterbank(n_fft, n_mels, fs)
    mel = fb @ power.T
    mel = np.log1p(mel)
    mel = mel - mel.min()
    return (mel / (mel.max() + 1e-8)).astype(np.float32)

def _prepare_acoustic_window(voltage: np.ndarray, fs: int) -> np.ndarray:
    x = voltage.astype(np.float64)
    x = x - np.mean(x)
    target = int(fs * 4.0)
    if len(x) >= target:
        start = (len(x) - target) // 2
        x = x[start : start + target]
    else:
        pad = target - len(x)
        x = np.pad(x, (pad // 2, pad - pad // 2))
    return x.astype(np.float32)

def fuse_cnn_probabilities(thermal_probs: dict, acoustic_probs: dict) -> list[dict]:
    """Fuse thermal + acoustic CNN softmax outputs (placeholder for fused SVM)."""
    fused = {}
    for code in CLASSES:
        tp = thermal_probs.get(code, 0.0)
        ap = acoustic_probs.get(code, 0.0)
        fused[code] = THERMAL_WEIGHT * tp + ACOUSTIC_WEIGHT * ap

    total = sum(fused.values()) or 1.0
    fused = {k: v / total for k, v in fused.items()}
    ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)
    return [{"label": CLASS_LABELS[code], "confidence": round(prob, 4)} for code, prob in ranked]

@torch.no_grad()
def thermal_probabilities(img: Image.Image, state: dict) -> dict:
    if not state or "classifier" not in state:
        return {}
    tensor = TF(img).unsqueeze(0).to(device)
    logits = state["classifier"](tensor)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    classes = state["classes"]
    return {c: float(probs[i]) for i, c in enumerate(classes)}

@torch.no_grad()
def acoustic_probabilities(voltage: np.ndarray, fs: int, state: dict) -> dict:
    if not state or state.get("acoustic_classifier") is None:
        return {}
    x = _prepare_acoustic_window(voltage, fs)
    mel = mel_spectrogram(x, fs)
    img = torch.from_numpy(mel).unsqueeze(0)
    img = transforms.Resize((224, 224), antialias=True)(img).repeat(3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    tensor = ((img - mean) / std).unsqueeze(0).float().to(device)

    logits = state["acoustic_classifier"](tensor)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    classes = state["acoustic_classes"]
    return {c: float(probs[i]) for i, c in enumerate(classes)}
