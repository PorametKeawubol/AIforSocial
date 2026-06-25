from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path

# RX 7600S / Navi 33 can be exposed as gfx1102, while some PyTorch ROCm
# wheels run these kernels through the gfx1100 target.
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"


def get_device() -> torch.device:
    # PyTorch ROCm exposes AMD GPUs through the same cuda API name.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_tags(value: object) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        return [str(item).strip() for item in ast.literal_eval(text) if str(item).strip()]
    return [item.strip() for item in text.split("|") if item.strip()]


def build_text(df: pd.DataFrame) -> list[str]:
    title = df["title"].fillna("").astype(str)
    description = df["description"].fillna("").astype(str)
    institute = df["institute"].fillna("").astype(str)
    return ("ชื่อวิชา: " + title + " | คำอธิบาย: " + description + " | สถาบัน: " + institute).tolist()


def load_train(labels: list[str]) -> tuple[list[str], np.ndarray]:
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    train_df["labels"] = train_df["tag"].apply(parse_tags)
    train_df["labels"] = train_df["labels"].apply(lambda row: [label for label in row if label in label_to_id])
    train_df = train_df[train_df["labels"].map(len) > 0].reset_index(drop=True)

    y = np.zeros((len(train_df), len(labels)), dtype=np.int32)
    for row_idx, row_labels in enumerate(train_df["labels"]):
        for label in row_labels:
            y[row_idx, label_to_id[label]] = 1
    return build_text(train_df), y


class TextDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, max_length: int) -> None:
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {key: value.squeeze(0) for key, value in encoded.items()}


def predict_probabilities(model, dataloader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            rows.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.vstack(rows)


def f1_for_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def tune_thresholds(y_true: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    thresholds = np.full(y_true.shape[1], 0.5, dtype=np.float32)
    for label_idx in range(y_true.shape[1]):
        best_f1 = -1.0
        best_threshold = 0.5
        for threshold in np.arange(0.15, 0.76, 0.05):
            y_pred = (probabilities[:, label_idx] >= threshold).astype(np.int32)
            score = f1_for_binary(y_true[:, label_idx], y_pred)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(threshold)
        thresholds[label_idx] = best_threshold
    return thresholds


def to_binary(probabilities: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    y_pred = (probabilities >= thresholds).astype(np.int32)
    empty_rows = np.where(y_pred.sum(axis=1) == 0)[0]
    if len(empty_rows):
        y_pred[empty_rows, probabilities[empty_rows].argmax(axis=1)] = 1
    return y_pred


def multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    micro_precision = tp / (tp + fp) if tp + fp else 0.0
    micro_recall = tp / (tp + fn) if tp + fn else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if micro_precision + micro_recall else 0.0

    per_label = []
    for idx in range(y_true.shape[1]):
        label_true = y_true[:, idx]
        label_pred = y_pred[:, idx]
        label_tp = ((label_true == 1) & (label_pred == 1)).sum()
        label_fp = ((label_true == 0) & (label_pred == 1)).sum()
        label_fn = ((label_true == 1) & (label_pred == 0)).sum()
        precision = label_tp / (label_tp + label_fp) if label_tp + label_fp else 0.0
        recall = label_tp / (label_tp + label_fn) if label_tp + label_fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label.append((precision, recall, f1, int(label_true.sum())))

    sample_scores = []
    for true_row, pred_row in zip(y_true, y_pred):
        intersection = ((true_row == 1) & (pred_row == 1)).sum()
        pred_count = pred_row.sum()
        true_count = true_row.sum()
        precision = intersection / pred_count if pred_count else 0.0
        recall = intersection / true_count if true_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        union = ((true_row == 1) | (pred_row == 1)).sum()
        jaccard = intersection / union if union else 0.0
        sample_scores.append((precision, recall, f1, jaccard))

    return {
        "exact_match_accuracy": float((y_true == y_pred).all(axis=1).mean()),
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "macro_precision": float(np.mean([row[0] for row in per_label])),
        "macro_recall": float(np.mean([row[1] for row in per_label])),
        "macro_f1": float(np.mean([row[2] for row in per_label])),
        "samples_precision": float(np.mean([row[0] for row in sample_scores])),
        "samples_recall": float(np.mean([row[1] for row in sample_scores])),
        "samples_f1": float(np.mean([row[2] for row in sample_scores])),
        "samples_jaccard": float(np.mean([row[3] for row in sample_scores])),
        "hamming_loss": float((y_true != y_pred).mean()),
        "per_label": per_label,
    }


def binary_to_tags(binary: np.ndarray, labels: list[str]) -> list[str]:
    return ["|".join(label for label, value in zip(labels, row) if value == 1) for row in binary]


def write_report(path: Path, metadata: dict, labels: list[str], thresholds: np.ndarray, metrics: dict[str, object], submission_path: Path) -> None:
    lines = [
        "# รายงานประสิทธิภาพโมเดล PhayaThaiBERT บน AMD GPU/ROCm",
        "",
        "## Model",
        "",
        f"- Base model: `{metadata['base_model']}`",
        "- Architecture: `AutoModelForSequenceClassification`",
        "- Task: multi-label classification",
        "- Activation: sigmoid",
        "- Loss: BCEWithLogitsLoss",
        f"- Learning rate: `{metadata['learning_rate']}`",
        f"- Epochs: `{metadata['epochs']}`",
        f"- Batch size: `{metadata['batch_size']}`",
        f"- Max length: `{metadata['max_length']}`",
        f"- Device during train: `{metadata.get('device')}`",
        f"- ROCm/HIP version: `{metadata.get('hip_version')}`",
        "",
        "หมายเหตุ: `test.csv` ไม่มีเฉลย `tag` จึงวัด metric จาก validation split ของ `train.csv` และสร้าง submission จาก `test.csv`",
        "",
        "## Validation Metrics",
        "",
        f"- Exact-match accuracy: {metrics['exact_match_accuracy']:.4f}",
        f"- Micro precision: {metrics['micro_precision']:.4f}",
        f"- Micro recall: {metrics['micro_recall']:.4f}",
        f"- Micro F1-score: {metrics['micro_f1']:.4f}",
        f"- Macro precision: {metrics['macro_precision']:.4f}",
        f"- Macro recall: {metrics['macro_recall']:.4f}",
        f"- Macro F1-score: {metrics['macro_f1']:.4f}",
        f"- Samples precision: {metrics['samples_precision']:.4f}",
        f"- Samples recall: {metrics['samples_recall']:.4f}",
        f"- Samples F1-score: {metrics['samples_f1']:.4f}",
        f"- Hamming loss: {metrics['hamming_loss']:.4f}",
        f"- Samples Jaccard score: {metrics['samples_jaccard']:.4f}",
        "",
        "## Per-label Metrics",
        "",
        "| tag | precision | recall | f1-score | support | threshold |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, threshold, row in zip(labels, thresholds, metrics["per_label"]):
        precision, recall, f1, support = row
        lines.append(f"| {label} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {support} | {threshold:.2f} |")
    lines += ["", "## Output", "", f"- Submission CSV: `{submission_path.relative_to(BASE_DIR)}`", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=str(BASE_DIR / "bert_rocm_model"))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--submission_path", default=str(DATA_DIR / "submission_bert_rocm.csv"))
    parser.add_argument("--report_path", default=str(BASE_DIR / "bert_rocm_performance_report.md"))
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    metadata_path = model_dir / "training_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"ไม่พบ {metadata_path} กรุณารัน train_bert.py ก่อน")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    labels = metadata["labels"]
    valid_idx = np.array(metadata["valid_indices"])
    max_length = int(metadata["max_length"])

    device = get_device()
    print(f"device: {device}")
    print(f"rocm/hip version: {getattr(torch.version, 'hip', None)}")
    print(f"HSA_OVERRIDE_GFX_VERSION: {os.environ.get('HSA_OVERRIDE_GFX_VERSION')}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)

    train_texts, y_all = load_train(labels)
    valid_dataset = TextDataset([train_texts[i] for i in valid_idx], tokenizer, max_length)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    y_valid = y_all[valid_idx]
    valid_probabilities = predict_probabilities(model, valid_loader, device)
    thresholds = tune_thresholds(y_valid, valid_probabilities)
    y_pred = to_binary(valid_probabilities, thresholds)
    metrics = multilabel_metrics(y_valid, y_pred)

    test_df = pd.read_csv(DATA_DIR / "test.csv")
    test_dataset = TextDataset(build_text(test_df), tokenizer, max_length)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    test_probabilities = predict_probabilities(model, test_loader, device)
    test_binary = to_binary(test_probabilities, thresholds)
    submission = pd.DataFrame({"index": test_df["index"], "tag": binary_to_tags(test_binary, labels)})
    submission_path = Path(args.submission_path)
    submission.to_csv(submission_path, index=False, encoding="utf-8-sig")

    report_path = Path(args.report_path)
    write_report(report_path, metadata, labels, thresholds, metrics, submission_path)

    print(f"saved submission: {submission_path}")
    print(f"saved report: {report_path}")
    print(f"exact-match accuracy: {metrics['exact_match_accuracy']:.4f}")
    print(f"micro f1: {metrics['micro_f1']:.4f}")
    print(f"macro f1: {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
