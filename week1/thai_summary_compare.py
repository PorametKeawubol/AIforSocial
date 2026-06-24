from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

os.environ.setdefault("HF_HOME", os.path.abspath(".hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModel, AutoTokenizer


TEXT = """
ประเทศไทยมีทรัพยากรธรรมชาติที่หลากหลาย ทั้งในด้านป่าไม้ แหล่งน้ำ และความหลากหลายทางชีวภาพ
เศรษฐกิจของประเทศอาศัยการส่งออก การท่องเที่ยว และภาคการเกษตรเป็นหลัก
ประเทศไทยกำลังเผชิญกับปัญหาสิ่งแวดล้อม เช่น มลพิษทางอากาศและการเปลี่ยนแปลงภูมิอากาศ
รัฐบาลจึงส่งเสริมนโยบาย BCG เพื่อความยั่งยืนทางเศรษฐกิจและสิ่งแวดล้อม
"""


@dataclass(frozen=True)
class ModelConfig:
    label: str
    model_id: str
    backend: str


MODELS = [
    ModelConfig(
        label="MiniLM multilingual sentence-transformer",
        model_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        backend="sentence-transformer",
    ),
    ModelConfig(
        label="WangchanBERTa",
        model_id="airesearch/wangchanberta-base-att-spm-uncased",
        backend="transformers",
    ),
    ModelConfig(
        label="PhayaThaiBERT",
        model_id="clicknext/phayathaibert",
        backend="transformers",
    ),
]


def split_sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in text.splitlines() if sentence.strip()]


def mean_pooling(model_output, attention_mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = model_output.last_hidden_state
    input_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask, dim=1) / torch.clamp(
        input_mask.sum(dim=1), min=1e-9
    )


def encode_with_transformers(model_id: str, sentences: list[str]) -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.eval()

    encoded = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        output = model(**encoded)

    embeddings = mean_pooling(output, encoded["attention_mask"])
    return F.normalize(embeddings, p=2, dim=1)


def summarize(config: ModelConfig, sentences: list[str], top_n: int) -> list[tuple[float, str]]:
    if config.backend == "sentence-transformer":
        model = SentenceTransformer(config.model_id)
        embeddings = model.encode(sentences, convert_to_tensor=True, normalize_embeddings=True)
    else:
        embeddings = encode_with_transformers(config.model_id, sentences)

    centroid = embeddings.mean(dim=0, keepdim=True)
    scores = util.pytorch_cos_sim(centroid, embeddings)[0]
    top_results = scores.topk(k=min(top_n, len(sentences)))

    return [
        (float(score), sentences[int(index)])
        for score, index in zip(top_results.values, top_results.indices)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Thai summarization by centroid similarity across 3 BERT-style models."
    )
    parser.add_argument("--top-n", type=int, default=2)
    args = parser.parse_args()

    sentences = split_sentences(TEXT)
    print(f"จำนวนประโยคทั้งหมด: {len(sentences)}")

    for config in MODELS:
        print()
        print(f"โมเดล: {config.label}")
        print(f"รหัสโมเดล: {config.model_id}")
        try:
            results = summarize(config, sentences, args.top_n)
        except Exception as exc:
            print(f"รันไม่สำเร็จ: {exc}")
            continue

        print("บทสรุป:")
        for score, sentence in results:
            print(f"- ({score:.4f}) {sentence}")


if __name__ == "__main__":
    main()
