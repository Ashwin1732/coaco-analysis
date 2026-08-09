#!/usr/bin/env python3
"""Cocoa thermal quality prediction API + frontend.

Upload a thermal image → predicted quality grade (CPB/OR/R/UR) plus
amplitude, frequency, and power (class-calibrated acoustic indicators,
optionally refined by an image→metrics regressor when available).
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEB_DIR / "static"
CLASS_CKPT = ROOT / "thermal_classifier" / "outputs" / "best_efficientnet_b0.pt"
ACOUSTIC_CKPT = ROOT / "acoustic_classifier" / "outputs" / "best_efficientnet_b0_acoustic.pt"
METRICS_CKPT = WEB_DIR / "outputs" / "metrics_regressor.joblib"
PROFILES_PATH = WEB_DIR / "outputs" / "class_metric_profiles.json"

CLASSES = ["CPB", "OR", "R", "UR"]
CLASS_LABELS = {
    "CPB": "Cocoa Pod Borer (infested)",
    "OR": "Over-ripe",
    "R": "Ripe",
    "UR": "Under-ripe",
}

app = FastAPI(title="Coaco Thermal Quality", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_state: dict = {"ready": False}


def build_classifier(num_classes: int) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def load_models() -> None:
    if not CLASS_CKPT.exists():
        raise FileNotFoundError(f"Missing classifier checkpoint: {CLASS_CKPT}")
    if not PROFILES_PATH.exists():
        raise FileNotFoundError(f"Missing class profiles: {PROFILES_PATH}")

    ckpt = torch.load(CLASS_CKPT, map_location=device, weights_only=False)
    classes = ckpt.get("classes", CLASSES)
    clf = build_classifier(len(classes))
    clf.load_state_dict(ckpt["model_state"])
    clf.to(device).eval()

    acoustic_clf = None
    acoustic_classes = []
    if ACOUSTIC_CKPT.exists():
        ackpt = torch.load(ACOUSTIC_CKPT, map_location=device, weights_only=False)
        acoustic_classes = ackpt.get("classes", ["CPB", "R", "UR"])
        acoustic_clf = build_classifier(len(acoustic_classes))
        acoustic_clf.load_state_dict(ackpt["model_state"])
        acoustic_clf.to(device).eval()

    profiles = json.loads(PROFILES_PATH.read_text())

    regressor = None
    backbone = None
    target_names = ["amplitude", "frequency", "power"]
    log1p = True
    if METRICS_CKPT.exists():
        bundle = joblib.load(METRICS_CKPT)
        regressor = bundle["regressor"]
        target_names = bundle.get("target_names", target_names)
        log1p = bundle.get("log1p", True)
        backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        backbone.classifier = nn.Identity()
        backbone.to(device).eval()

    _state.update(
        {
            "ready": True,
            "classes": classes,
            "classifier": clf,
            "acoustic_classes": acoustic_classes,
            "acoustic_classifier": acoustic_clf,
            "profiles": profiles,
            "backbone": backbone,
            "regressor": regressor,
            "target_names": target_names,
            "log1p": log1p,
        }
    )


TF = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def read_image(data: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc


FS = 12800
N_SAMPLES = int(FS * 4.0)

def load_signal_from_bytes(data: bytes) -> np.ndarray:
    vals: list[float] = []
    reader = csv.reader(io.StringIO(data.decode("utf-8")))
    next(reader, None)
    for row in reader:
        if len(row) >= 2:
            try:
                vals.append(float(row[1]))
            except ValueError:
                continue
    x = np.asarray(vals, dtype=np.float64)
    if len(x) == 0:
        raise ValueError("Empty signal")
    x = x - np.mean(x)
    if len(x) >= N_SAMPLES:
        start = (len(x) - N_SAMPLES) // 2
        x = x[start : start + N_SAMPLES]
    else:
        pad = N_SAMPLES - len(x)
        x = np.pad(x, (pad // 2, pad - pad // 2))
    return x.astype(np.float32)


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
            if 0 <= j < fb.shape[1]:
                fb[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if 0 <= j < fb.shape[1]:
                fb[i, j] = (right - j) / (right - center)
    return fb


def mel_spectrogram(x: np.ndarray, n_fft: int = 1024, hop: int = 256, n_mels: int = 128) -> np.ndarray:
    window = np.hanning(n_fft).astype(np.float32)
    if len(x) < n_fft: x = np.pad(x, (0, n_fft - len(x)))
    n_frames = 1 + (len(x) - n_fft) // hop
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(n_frames, n_fft), strides=(x.strides[0] * hop, x.strides[0]), writeable=False
    ).copy()
    frames *= window
    spec = np.fft.rfft(frames, axis=1)
    power = (spec.real**2 + spec.imag**2).astype(np.float32)
    fb = mel_filterbank(n_fft, n_mels, FS)
    mel = fb @ power.T
    mel = np.log1p(mel)
    mel = mel - mel.min()
    denom = mel.max() + 1e-8
    return (mel / denom).astype(np.float32)


def acoustic_metrics(x: np.ndarray) -> dict[str, float]:
    amp = float(np.sqrt(np.mean(x**2)))
    power = float(np.mean(x**2))
    n_fft, hop = 2048, 512
    window = np.hanning(n_fft)
    if len(x) < n_fft: x = np.pad(x, (0, n_fft - len(x)))
    n_frames = 1 + (len(x) - n_fft) // hop
    cents = []
    freqs = np.fft.rfftfreq(n_fft, 1.0 / FS)
    for i in range(n_frames):
        frame = x[i * hop : i * hop + n_fft] * window
        ps = np.abs(np.fft.rfft(frame)) ** 2
        cents.append(float((freqs * ps).sum() / (ps.sum() + 1e-12)))
    freq = float(np.mean(cents)) if cents else 0.0
    return {"amplitude": amp, "frequency": freq, "power": power}


@torch.no_grad()
def predict_acoustic(csv_bytes: bytes) -> dict:
    if not _state["ready"]: load_models()
    if _state["acoustic_classifier"] is None:
        raise RuntimeError("Acoustic classifier model not found")

    x = load_signal_from_bytes(csv_bytes)
    metrics = acoustic_metrics(x)

    mel = mel_spectrogram(x)
    img = torch.from_numpy(mel).unsqueeze(0)
    img = transforms.Resize((224, 224), antialias=True)(img)
    img = img.repeat(3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    img = (img - mean) / std
    tensor = img.unsqueeze(0).float().to(device)

    logits = _state["acoustic_classifier"](tensor)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    classes = _state["acoustic_classes"]
    idx = int(probs.argmax())
    quality = classes[idx]

    return {
        "quality": quality,
        "quality_label": CLASS_LABELS.get(quality, quality),
        "confidence": float(probs[idx]),
        "class_probabilities": {c: float(probs[i]) for i, c in enumerate(classes)},
        "amplitude": {"value": metrics["amplitude"], "unit": "RMS"},
        "frequency": {"value": metrics["frequency"], "unit": "Hz"},
        "power": {"value": metrics["power"], "unit": "mean-square"},
        "metrics_source": "extracted from signal",
    }


@torch.no_grad()
def predict_all(img: Image.Image) -> dict:
    if not _state["ready"]:
        load_models()

    tensor = TF(img).unsqueeze(0).to(device)
    logits = _state["classifier"](tensor)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    classes = _state["classes"]
    idx = int(probs.argmax())
    quality = classes[idx]
    confidence = float(probs[idx])

    # Class-calibrated acoustic quality indicators
    profile = _state["profiles"][quality]
    amplitude = float(profile["amplitude"]["value"])
    frequency = float(profile["frequency"]["value"])
    power = float(profile["power"]["value"])
    source = "class_profile"

    # Soft blend with image regressor when available (confidence-weighted)
    if _state["regressor"] is not None and _state["backbone"] is not None:
        emb = _state["backbone"](tensor).cpu().numpy()
        pred = _state["regressor"].predict(emb)[0]
        if _state["log1p"]:
            pred = np.expm1(pred)
        pred = np.maximum(pred, 0.0)
        w = 0.35  # keep class profile dominant; regressor is weak alone
        amplitude = (1 - w) * amplitude + w * float(pred[0])
        frequency = (1 - w) * frequency + w * float(pred[1])
        power = (1 - w) * power + w * float(pred[2])
        source = "class_profile+image_regressor"

    return {
        "quality": quality,
        "quality_label": CLASS_LABELS.get(quality, quality),
        "confidence": confidence,
        "class_probabilities": {c: float(probs[i]) for i, c in enumerate(classes)},
        "amplitude": {
            "value": amplitude,
            "unit": profile["amplitude"]["unit"],
        },
        "frequency": {
            "value": frequency,
            "unit": profile["frequency"]["unit"],
        },
        "power": {
            "value": power,
            "unit": profile["power"]["unit"],
        },
        "metrics_source": source,
    }


@app.on_event("startup")
def startup() -> None:
    try:
        load_models()
        print(f"Models loaded on {device}", flush=True)
    except Exception as exc:
        print(f"Startup warning: {exc}", flush=True)


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {
        "ok": _state.get("ready", False),
        "device": str(device),
        "classifier": CLASS_CKPT.exists(),
        "profiles": PROFILES_PATH.exists(),
        "regressor": METRICS_CKPT.exists(),
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type:
        raise HTTPException(status_code=400, detail="Missing content type")
    
    is_csv = file.filename.lower().endswith(".csv") or "csv" in file.content_type
    is_image = file.content_type.startswith("image/")
    
    if not (is_image or is_csv):
        raise HTTPException(status_code=400, detail="Please upload an image or CSV file")
        
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
        
    try:
        if is_csv:
            result = predict_acoustic(data)
        else:
            img = read_image(data)
            result = predict_all(img)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    result["filename"] = file.filename
    return result
