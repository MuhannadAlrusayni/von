"""Dataset preparation pipeline for training Von's native decision models.

Pulls ANLI (Rounds 1, 2, 3) and WANLI, formats into Von System One Choice/Noul
records, and exports balanced train.jsonl and val.jsonl.
"""

import argparse
import json
import os
import random
from typing import Dict, List, Optional
from datasets import load_dataset


def map_anli_row(row: dict) -> Optional[dict]:
    premise = (row.get("premise") or "").strip()
    hypothesis = (row.get("hypothesis") or "").strip()
    label_num = row.get("label")

    if not premise or not hypothesis or label_num not in (0, 1, 2):
        return None

    # ANLI label mapping: 0 -> entailment, 1 -> neutral, 2 -> contradiction
    label_map = {0: "supported", 1: "insufficient", 2: "contradicted"}
    target = label_map[label_num]

    return {
        "state": premise,
        "question": f"Is the following claim supported, contradicted, or is evidence insufficient: '{hypothesis}'?",
        "options": [
            {"id": "supported", "description": f"The evidence establishes that: {hypothesis}."},
            {"id": "contradicted", "description": f"The evidence contradicts that: {hypothesis}."},
            {"id": "insufficient", "description": f"The evidence is insufficient to verify: {hypothesis}."},
        ],
        "label": target,
        "source": "anli",
    }


def map_wanli_row(row: dict) -> Optional[dict]:
    premise = (row.get("premise") or "").strip()
    hypothesis = (row.get("hypothesis") or "").strip()
    gold = (row.get("gold") or "").lower().strip()

    if not premise or not hypothesis:
        return None

    label_map = {
        "entailment": "supported",
        "neutral": "insufficient",
        "contradiction": "contradicted",
    }
    target = label_map.get(gold)
    if not target:
        return None

    return {
        "state": premise,
        "question": f"Is the following claim supported, contradicted, or is evidence insufficient: '{hypothesis}'?",
        "options": [
            {"id": "supported", "description": f"The evidence establishes that: {hypothesis}."},
            {"id": "contradicted", "description": f"The evidence contradicts that: {hypothesis}."},
            {"id": "insufficient", "description": f"The evidence is insufficient to verify: {hypothesis}."},
        ],
        "label": target,
        "source": "wanli",
    }


def build_corpus(
    output_dir: str = "data",
    max_train_samples: int = 100000,
    val_samples: int = 3000,
    seed: int = 42,
):
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    print("Fetching ANLI (R1, R2, R3)...")
    anli_r1 = load_dataset("facebook/anli", split="train_r1")
    anli_r2 = load_dataset("facebook/anli", split="train_r2")
    anli_r3 = load_dataset("facebook/anli", split="train_r3")

    print("Fetching WANLI...")
    wanli_train = load_dataset("alisawuffles/WANLI", split="train")

    records: List[dict] = []

    print("Processing ANLI rows...")
    for split in [anli_r1, anli_r2, anli_r3]:
        for row in split:
            rec = map_anli_row(row)
            if rec:
                records.append(rec)

    print(f"Loaded {len(records)} ANLI records.")

    print("Processing WANLI rows...")
    for row in wanli_train:
        rec = map_wanli_row(row)
        if rec:
            records.append(rec)

    print(f"Total raw candidates collected: {len(records)}")

    # Shuffle
    random.shuffle(records)

    # Balance classes (supported, contradicted, insufficient)
    by_label: Dict[str, List[dict]] = {"supported": [], "contradicted": [], "insufficient": []}
    for r in records:
        lbl = r["label"]
        if lbl in by_label:
            by_label[lbl].append(r)

    print(f"Class distribution: supported={len(by_label['supported'])}, contradicted={len(by_label['contradicted'])}, insufficient={len(by_label['insufficient'])}")

    min_count = min(len(v) for v in by_label.values())
    per_class_limit = min(min_count, (max_train_samples + val_samples) // 3)

    balanced: List[dict] = []
    for lbl in ["supported", "contradicted", "insufficient"]:
        balanced.extend(by_label[lbl][:per_class_limit])

    random.shuffle(balanced)

    val_set = balanced[:val_samples]
    train_set = balanced[val_samples : val_samples + max_train_samples]

    train_path = os.path.join(output_dir, "train.jsonl")
    val_path = os.path.join(output_dir, "val.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_set:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for item in val_set:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nCorpus build complete:")
    print(f"  -> Train set: {len(train_set)} rows saved to {train_path}")
    print(f"  -> Val set:   {len(val_set)} rows saved to {val_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build training corpus from ANLI + WANLI")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--max_train", type=int, default=100000)
    parser.add_argument("--val_samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_corpus(
        output_dir=args.output_dir,
        max_train_samples=args.max_train,
        val_samples=args.val_samples,
        seed=args.seed,
    )
