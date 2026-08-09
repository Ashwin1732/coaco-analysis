#!/usr/bin/env python3
"""3-class cocoa acoustic classifier (CPB / R / UR).

OR excluded (only 1 recording). Pod-level stratified 80/20 split so
replicate taps of the same pod never leak across train and test.

Trains:
  1) Classical model on handcrafted features (RandomForest)
  2) EfficientNet-B0 on mel spectrograms (GPU)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

CLASSES = ["CPB", "R", "UR"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
POD_RE = re.compile(r"^(CPB|R|UR)(\d+)", re.I)
FS = 12800
DURATION_S = 4.0
N_SAMPLES = int(FS * DURATION_S)  # 51200


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_signal(path: Path) -> np.ndarray:
    vals: list[float] = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                try:
                    vals.append(float(row[1]))
                except ValueError:
                    continue
    x = np.asarray(vals, dtype=np.float64)
    if len(x) == 0:
        raise ValueError(f"Empty signal: {path}")
    # center
    x = x - np.mean(x)
    # pad / truncate to fixed length
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
        if center == left:
            center += 1
        if right == center:
            right += 1
        for j in range(left, center):
            if 0 <= j < fb.shape[1]:
                fb[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if 0 <= j < fb.shape[1]:
                fb[i, j] = (right - j) / (right - center)
    return fb


def mel_spectrogram(x: np.ndarray, n_fft: int = 1024, hop: int = 256, n_mels: int = 128) -> np.ndarray:
    # Hann-windowed STFT power
    window = np.hanning(n_fft).astype(np.float32)
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    n_frames = 1 + (len(x) - n_fft) // hop
    frames = np.lib.stride_tricks.as_strided(
        x,
        shape=(n_frames, n_fft),
        strides=(x.strides[0] * hop, x.strides[0]),
        writeable=False,
    ).copy()
    frames *= window
    spec = np.fft.rfft(frames, axis=1)
    power = (spec.real**2 + spec.imag**2).astype(np.float32)
    fb = mel_filterbank(n_fft, n_mels, FS)
    mel = fb @ power.T  # (n_mels, n_frames)
    mel = np.log1p(mel)
    # per-sample normalize to 0-1 for imaging
    mel = mel - mel.min()
    denom = mel.max() + 1e-8
    mel = mel / denom
    return mel.astype(np.float32)


def handcrafted_features(x: np.ndarray) -> np.ndarray:
    rms = float(np.sqrt(np.mean(x**2)))
    peak = float(np.max(np.abs(x)))
    crest = peak / (rms + 1e-12)
    zcr = float(np.mean(np.abs(np.diff(np.sign(x)))) / 2.0)
    # spectral features from mel
    mel = mel_spectrogram(x, n_mels=40)
    # band energy shares on mel axis
    bands = np.array_split(mel, 6, axis=0)
    band_e = np.array([b.sum() for b in bands], dtype=np.float64)
    band_e = band_e / (band_e.sum() + 1e-12)
    centroid = float((np.arange(mel.shape[0])[:, None] * mel).sum() / (mel.sum() + 1e-12))
    flatness = float(np.exp(np.mean(np.log(mel + 1e-12))) / (np.mean(mel) + 1e-12))
    # MFCC-ish: DCT of mean mel
    mean_mel = mel.mean(axis=1)
    # Type-II DCT (MFCC-style) without requiring scipy.fftpack at import time
    n = len(mean_mel)
    k = np.arange(n)[:, None]
    n_idx = np.arange(n)[None, :]
    dct_mat = np.cos(np.pi / n * (n_idx + 0.5) * k)
    mfcc = (dct_mat @ mean_mel)[:13]
    stats = np.array(
        [
            rms,
            peak,
            crest,
            zcr,
            centroid,
            flatness,
            float(np.mean(x**3) / ((np.mean(x**2) + 1e-12) ** 1.5)),  # skew
            float(np.mean(x**4) / ((np.mean(x**2) + 1e-12) ** 2)),  # kurt
            *band_e,
            *mfcc,
        ],
        dtype=np.float64,
    )
    return stats


def discover_pods(data_root: Path) -> dict[str, list[dict]]:
    by_class: dict[str, dict[str, list[Path]]] = {c: defaultdict(list) for c in CLASSES}
    for cls in CLASSES:
        folder = data_root / cls
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing class folder: {folder}")
        for path in sorted(folder.glob("*.csv")):
            m = POD_RE.match(path.name)
            if not m:
                continue
            label, pod_num = m.group(1).upper(), m.group(2)
            if label != cls:
                continue
            by_class[cls][f"{cls}{pod_num}"].append(path)
    pods: dict[str, list[dict]] = {}
    for cls in CLASSES:
        pods[cls] = [
            {"pod_id": pod_id, "paths": paths}
            for pod_id, paths in sorted(by_class[cls].items())
        ]
        if not pods[cls]:
            raise RuntimeError(f"No CSV files for class {cls}")
    return pods


def pod_level_split(
    pods: dict[str, list[dict]], test_size: float, seed: int
) -> tuple[list[dict], list[dict]]:
    train_records: list[dict] = []
    test_records: list[dict] = []
    for cls, pod_list in pods.items():
        labels = [cls] * len(pod_list)
        train_pods, test_pods = train_test_split(
            pod_list,
            test_size=test_size,
            random_state=seed,
            stratify=labels if len(pod_list) >= 2 else None,
        )
        for split, bucket in ((train_pods, train_records), (test_pods, test_records)):
            for pod in split:
                for path in pod["paths"]:
                    bucket.append(
                        {
                            "path": path,
                            "label": CLASS_TO_IDX[cls],
                            "class": cls,
                            "pod_id": pod["pod_id"],
                        }
                    )
    return train_records, test_records


class SpecDataset(Dataset):
    def __init__(self, records: list[dict], augment: bool = False):
        self.records = records
        self.augment = augment
        self.resize = transforms.Resize((224, 224), antialias=True)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        x = load_signal(rec["path"])
        if self.augment:
            # time shift
            shift = random.randint(-FS // 4, FS // 4)
            x = np.roll(x, shift)
            # gain
            x = x * random.uniform(0.8, 1.2)
        mel = mel_spectrogram(x)  # (n_mels, frames)
        img = torch.from_numpy(mel).unsqueeze(0)  # 1, H, W
        img = self.resize(img)
        img = img.repeat(3, 1, 1)  # fake RGB for EfficientNet
        # ImageNet-ish normalize
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        img = (img - mean) / std
        return img.float(), rec["label"]


def build_cnn(num_classes: int, device: torch.device) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model.to(device)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate_cnn(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    y_true, y_pred = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
    return (
        total_loss / max(total, 1),
        correct / max(total, 1),
        np.asarray(y_true),
        np.asarray(y_pred),
    )


def train_random_forest(train_records, test_records, seed: int):
    print("Extracting handcrafted features...", flush=True)
    X_train, y_train = [], []
    for rec in train_records:
        X_train.append(handcrafted_features(load_signal(rec["path"])))
        y_train.append(rec["label"])
    X_test, y_test = [], []
    for rec in test_records:
        X_test.append(handcrafted_features(load_signal(rec["path"])))
        y_test.append(rec["label"])
    X_train, y_train = np.asarray(X_train), np.asarray(y_train)
    X_test, y_test = np.asarray(X_test), np.asarray(y_test)

    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=None,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = float((pred == y_test).mean())
    report = classification_report(
        y_test, pred, target_names=CLASSES, digits=4, output_dict=True
    )
    cm = confusion_matrix(y_test, pred).tolist()
    print(f"RandomForest test_acc={acc:.4f}", flush=True)
    print(classification_report(y_test, pred, target_names=CLASSES, digits=4), flush=True)
    return {"test_acc": acc, "report": report, "confusion_matrix": cm, "model": clf}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "Acoustics_Extracted"
        / "Acoustics",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "auto"])
    args = parser.parse_args()

    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")

    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Data root: {args.data_root}", flush=True)
    print("Classes: CPB / R / UR (OR excluded — only 1 CSV)", flush=True)

    pods = discover_pods(args.data_root)
    train_records, test_records = pod_level_split(pods, args.test_size, args.seed)
    train_pods = sorted({r["pod_id"] for r in train_records})
    test_pods = sorted({r["pod_id"] for r in test_records})
    assert not (set(train_pods) & set(test_pods))

    for cls in CLASSES:
        n_tr = sum(1 for p in train_pods if p.startswith(cls))
        n_te = sum(1 for p in test_pods if p.startswith(cls))
        print(f"  {cls}: train_pods={n_tr} test_pods={n_te}", flush=True)
    print(
        f"Files: train={len(train_records)} test={len(test_records)} "
        f"(train_pods={len(train_pods)} test_pods={len(test_pods)})",
        flush=True,
    )
    print("Train counts:", Counter(r["class"] for r in train_records), flush=True)
    print("Test counts:", Counter(r["class"] for r in test_records), flush=True)

    # ---- RandomForest baseline ----
    rf_result = train_random_forest(train_records, test_records, args.seed)
    import joblib

    joblib.dump(rf_result["model"], args.out_dir / "random_forest.joblib")

    # ---- Spectrogram CNN ----
    train_ds = SpecDataset(train_records, augment=True)
    test_ds = SpecDataset(test_records, augment=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_cnn(len(CLASSES), device)
    # class weights for imbalance (R has fewer files)
    train_labels = [r["label"] for r in train_records]
    counts = np.bincount(train_labels, minlength=len(CLASSES)).astype(np.float64)
    weights = counts.sum() / (len(CLASSES) * np.maximum(counts, 1.0))
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    best_acc = -1.0
    best_path = args.out_dir / "best_efficientnet_b0_acoustic.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc, _, _ = evaluate_cnn(model, test_loader, criterion, device)
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
            }
        )
        print(
            f"Epoch {epoch:02d}/{args.epochs}  "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f}  "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.3f}",
            flush=True,
        )
        if test_acc >= best_acc:
            best_acc = test_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": CLASSES,
                    "epoch": epoch,
                    "test_acc": test_acc,
                },
                best_path,
            )

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    _, cnn_acc, y_true, y_pred = evaluate_cnn(model, test_loader, criterion, device)
    cnn_report = classification_report(
        y_true, y_pred, target_names=CLASSES, digits=4, output_dict=True
    )
    cnn_cm = confusion_matrix(y_true, y_pred).tolist()
    report_text = classification_report(y_true, y_pred, target_names=CLASSES, digits=4)

    print("\n=== Best CNN ===", flush=True)
    print(f"Checkpoint: {best_path}", flush=True)
    print(f"Best epoch: {ckpt['epoch']}  test_acc={cnn_acc:.4f}", flush=True)
    print(report_text, flush=True)
    print("Confusion matrix:", flush=True)
    print(np.array(cnn_cm), flush=True)

    metrics = {
        "device": str(device),
        "classes": CLASSES,
        "note": "OR excluded (n=1)",
        "split": {
            "strategy": "pod-level stratified 80/20",
            "test_size": args.test_size,
            "seed": args.seed,
            "train_pods": len(train_pods),
            "test_pods": len(test_pods),
            "train_files": len(train_records),
            "test_files": len(test_records),
        },
        "random_forest": {
            "test_acc": rf_result["test_acc"],
            "classification_report": rf_result["report"],
            "confusion_matrix": rf_result["confusion_matrix"],
        },
        "efficientnet_b0": {
            "best_epoch": ckpt["epoch"],
            "test_acc": cnn_acc,
            "classification_report": cnn_report,
            "confusion_matrix": cnn_cm,
            "history": history,
        },
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    (args.out_dir / "classification_report_cnn.txt").write_text(report_text)
    (args.out_dir / "classification_report_rf.txt").write_text(
        classification_report(
            [r["label"] for r in test_records],
            rf_result["model"].predict(
                np.asarray(
                    [handcrafted_features(load_signal(r["path"])) for r in test_records]
                )
            ),
            target_names=CLASSES,
            digits=4,
        )
    )

    winner = (
        "RandomForest"
        if rf_result["test_acc"] >= cnn_acc
        else "EfficientNet-B0 (mel spectrogram)"
    )
    print(
        f"\nSummary: RF={rf_result['test_acc']:.4f}  CNN={cnn_acc:.4f}  "
        f"best={winner}",
        flush=True,
    )


if __name__ == "__main__":
    main()
