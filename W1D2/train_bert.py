from __future__ import annotations

import argparse
import ast
import json
import os
import random
from pathlib import Path

# RX 7600S / Navi 33 can be exposed as gfx1102, while some PyTorch ROCm
# wheels run these kernels through the gfx1100 target.
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
DEFAULT_MODEL = "clicknext/phayathaibert"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    # PyTorch ROCm exposes AMD GPUs through the same cuda API name.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_device_info(device: torch.device) -> None:
    print(f"device: {device}")
    print(f"torch version: {torch.__version__}")
    print(f"rocm/hip version: {getattr(torch.version, 'hip', None)}")
    print(f"HSA_OVERRIDE_GFX_VERSION: {os.environ.get('HSA_OVERRIDE_GFX_VERSION')}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    else:
        print("AMD GPU is not active. Install PyTorch ROCm and ROCm drivers, then run again.")


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


def split_indices(n_rows: int, valid_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n_rows)
    rng.shuffle(indices)
    n_valid = max(1, int(round(n_rows * valid_size)))
    return np.sort(indices[n_valid:]), np.sort(indices[:n_valid])


def load_data(valid_size: float, seed: int) -> tuple[list[str], np.ndarray, list[str], np.ndarray, np.ndarray]:
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    train_df["labels"] = train_df["tag"].apply(parse_tags)

    label_counts = pd.Series([label for labels in train_df["labels"] for label in labels]).value_counts()
    labels = sorted(label_counts[label_counts >= 2].index.tolist())
    label_to_id = {label: idx for idx, label in enumerate(labels)}

    train_df["labels"] = train_df["labels"].apply(lambda row: [label for label in row if label in label_to_id])
    train_df = train_df[train_df["labels"].map(len) > 0].reset_index(drop=True)

    y = np.zeros((len(train_df), len(labels)), dtype=np.float32)
    for row_idx, row_labels in enumerate(train_df["labels"]):
        for label in row_labels:
            y[row_idx, label_to_id[label]] = 1.0

    train_idx, valid_idx = split_indices(len(train_df), valid_size, seed)
    return build_text(train_df), y, labels, train_idx, valid_idx


class ThaiMoocDataset(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray, tokenizer, max_length: int) -> None:
        self.texts = texts
        self.labels = labels.astype(np.float32)
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
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item


def evaluate_loss(model, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            losses.append(float(loss.detach().cpu()))
    model.train()
    return float(np.mean(losses)) if losses else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument("--output_dir", default=str(BASE_DIR / "bert_rocm_model"))
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--valid_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print_device_info(device)

    texts, y, labels, train_idx, valid_idx = load_data(args.valid_size, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(labels),
        problem_type="multi_label_classification",
        id2label={idx: label for idx, label in enumerate(labels)},
        label2id={label: idx for idx, label in enumerate(labels)},
    ).to(device)

    train_dataset = ThaiMoocDataset([texts[i] for i in train_idx], y[train_idx], tokenizer, args.max_length)
    valid_dataset = ThaiMoocDataset([texts[i] for i in valid_idx], y[valid_idx], tokenizer, args.max_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    history = []
    model.train()
    for epoch in range(1, args.epochs + 1):
        train_losses = []
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_losses.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(train_losses))
        valid_loss = evaluate_loss(model, valid_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss})
        print(f"epoch {epoch}/{args.epochs} - train_loss={train_loss:.4f} valid_loss={valid_loss:.4f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "base_model": args.model_name,
        "device": str(device),
        "torch_version": torch.__version__,
        "hip_version": getattr(torch.version, "hip", None),
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "valid_size": args.valid_size,
        "seed": args.seed,
        "labels": labels,
        "train_indices": train_idx.tolist(),
        "valid_indices": valid_idx.tolist(),
        "history": history,
    }
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved model: {output_dir}")


if __name__ == "__main__":
    main()
