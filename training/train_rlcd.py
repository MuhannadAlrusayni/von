"""Train Von's ModernBERT native decision model with RLCD calibration.

Features:
- Backbone: ModernBERT-Large
- Loss: Cross-Entropy + Brier Score Calibration
- Automatic Post-Hoc Temperature Optimization
- Mixed Precision (bfloat16 / float16)
- Checkpoint export ready for Von / Hugging Face
"""

import argparse
import json
import math
import os
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup


class DecisionDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.rows = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


def collate_fn(batch: List[dict], tokenizer, max_length: int = 512):
    premises = []
    hypotheses = []
    labels = []
    option_counts = []

    label_to_idx = {"supported": 0, "contradicted": 1, "insufficient": 2}

    for item in batch:
        state = item["state"]
        q = item["question"]
        opts = item["options"]
        target = item["label"]

        # Track target index within options list
        opt_ids = [opt["id"] for opt in opts]
        if target in opt_ids:
            target_idx = opt_ids.index(target)
        else:
            target_idx = 0

        labels.append(target_idx)
        option_counts.append(len(opts))

        for opt in opts:
            premises.append(state)
            hypotheses.append(f"{q} {opt['description']}")

    encodings = tokenizer(
        premises,
        hypotheses,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    return {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": torch.tensor(labels, dtype=torch.long),
        "option_counts": option_counts,
    }


def compute_rlcd_loss(
    entail_scores: torch.Tensor,
    labels: torch.Tensor,
    option_counts: List[int],
    brier_weight: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Calculate Cross-Entropy + Brier calibration loss over candidate option pools."""
    offset = 0
    ce_losses = []
    brier_losses = []
    correct = 0
    total = len(labels)

    for i, count in enumerate(option_counts):
        scores = entail_scores[offset : offset + count]  # shape: (K,)
        target_idx = labels[i].item()

        probs = torch.softmax(scores, dim=-1)

        # 1. Cross Entropy Loss: -log(p_target)
        p_target = torch.clamp(probs[target_idx], min=1e-7, max=1.0)
        ce_loss = -torch.log(p_target)
        ce_losses.append(ce_loss)

        # 2. Brier Score Loss: sum((p_k - y_k)^2)
        one_hot = torch.zeros_like(probs)
        one_hot[target_idx] = 1.0
        brier = torch.sum((probs - one_hot) ** 2)
        brier_losses.append(brier)

        if torch.argmax(probs).item() == target_idx:
            correct += 1

        offset += count

    total_ce = torch.stack(ce_losses).mean()
    total_brier = torch.stack(brier_losses).mean()
    total_loss = total_ce + brier_weight * total_brier
    accuracy = correct / total if total > 0 else 0.0

    return total_loss, total_ce, accuracy


def fit_temperature(all_scores: List[torch.Tensor], all_labels: List[int]) -> float:
    """Optimize scalar temperature T on validation split to minimize Negative Log Likelihood."""
    from scipy.optimize import minimize_scalar

    def nll_eval(temp_val: float) -> float:
        t = max(temp_val, 0.01)
        loss = 0.0
        for scores, target in zip(all_scores, all_labels):
            probs = torch.softmax(scores / t, dim=-1)
            p_target = torch.clamp(probs[target], min=1e-7, max=1.0)
            loss += -math.log(p_target.item())
        return loss / len(all_labels)

    res = minimize_scalar(nll_eval, bounds=(0.5, 3.0), method="bounded")
    return float(res.x)


def train(
    train_path: str = "data/train.jsonl",
    val_path: str = "data/val.jsonl",
    model_id: str = "tasksource/ModernBERT-large-nli",
    output_dir: str = "checkpoints/von-modernbert-rlcd",
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 2e-5,
    brier_weight: float = 0.5,
    max_length: int = 512,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading tokenizer & model: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device)

    # Detect entailment index in id2label
    entail_idx = 0
    id2label = getattr(model.config, "id2label", {})
    for idx, lbl in id2label.items():
        if "entail" in lbl.lower():
            entail_idx = int(idx)
            break
    print(f"Detected entailment class index: {entail_idx} ({id2label.get(entail_idx)})")

    train_ds = DecisionDataset(train_path)
    val_ds = DecisionDataset(val_path)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, max_length=max_length),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, max_length=max_length),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    print(f"\nStarting training:")
    print(f"  -> Total Train Samples: {len(train_ds)}")
    print(f"  -> Validation Samples:  {len(val_ds)}")
    print(f"  -> Epochs:              {epochs}")
    print(f"  -> Effective Steps:     {total_steps}\n")

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            option_counts = batch["option_counts"]

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                entail_scores = logits[:, entail_idx]
                loss, ce_loss, acc = compute_rlcd_loss(
                    entail_scores, labels, option_counts, brier_weight=brier_weight
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_acc += acc

            if (step + 1) % 50 == 0 or (step + 1) == len(train_loader):
                elapsed = time.time() - t0
                print(
                    f"Epoch [{epoch}/{epochs}] Step [{step+1}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f} (CE: {ce_loss.item():.4f}) Acc: {acc*100:.1f}% "
                    f"Elapsed: {elapsed:.1f}s"
                )

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_scores_list = []
        val_labels_list = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                option_counts = batch["option_counts"]

                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                    entail_scores = logits[:, entail_idx]
                    loss, _, acc = compute_rlcd_loss(
                        entail_scores, labels, option_counts, brier_weight=brier_weight
                    )

                val_loss += loss.item()
                val_acc += acc

                # Collect for temperature fitting
                offset = 0
                for i, cnt in enumerate(option_counts):
                    val_scores_list.append(entail_scores[offset : offset + cnt].cpu())
                    val_labels_list.append(labels[i].item())
                    offset += cnt

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)
        print(f"\n--- Epoch {epoch} Validation: Loss = {avg_val_loss:.4f}, Accuracy = {avg_val_acc*100:.2f}% ---\n")

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            print(f"Saving new best checkpoint to {output_dir}...")
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

    print("\nFitting post-hoc calibration temperature (T)...")
    opt_temp = fit_temperature(val_scores_list, val_labels_list)
    print(f"Optimal fitted temperature: T = {opt_temp:.4f}")

    calibration_config = {
        "base_model": model_id,
        "temperature": round(opt_temp, 4),
        "best_val_accuracy": round(best_val_acc, 4),
        "timestamp": time.time(),
    }
    with open(os.path.join(output_dir, "calibration.json"), "w") as f:
        json.dump(calibration_config, f, indent=2)

    print(f"\nTraining & calibration complete! Model exported to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Von ModernBERT with RLCD")
    parser.add_argument("--train_data", type=str, default="data/train.jsonl")
    parser.add_argument("--val_data", type=str, default="data/val.jsonl")
    parser.add_argument("--model_id", type=str, default="tasksource/ModernBERT-large-nli")
    parser.add_argument("--output_dir", type=str, default="checkpoints/von-modernbert-rlcd")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--brier_weight", type=float, default=0.5)
    args = parser.parse_args()

    train(
        train_path=args.train_data,
        val_path=args.val_data,
        model_id=args.model_id,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        brier_weight=args.brier_weight,
    )
