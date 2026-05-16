"""
Argument Quality Model — Fine-tuning DeBERTa-v3-small on Args.me corpus

Architecture:
    DeBERTa-v3-small  →  [CLS] pooling  →  Dropout  →  Linear(hidden, 1)  →  Sigmoid

Saved artefacts (./quality_model/ by default):
    config.json        — hyperparameters used for final training
    pytorch_model.bin  — model weights
    tokenizer/         — HuggingFace tokenizer files
    test_metrics.json  — held-out test results

Usage:
    # Full pipeline (HPO → retrain → test):
    python train_quality_model.py \\
        --args-data  webis-argquality20-full.csv \\
        --topic-data webis-argquality20-topics.csv \\
        --output     ./quality_model

    # Skip HPO, use default hyperparameters:
    python train_quality_model.py ... --no-hpo

    # Evaluate existing checkpoint on test set only:
    python train_quality_model.py ... --eval-only
"""

import argparse
import json
import logging
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from scipy.stats import pearsonr
import optuna

log = logging.getLogger("train_quality_model")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("train_quality_model.log", mode="w", encoding="utf-8"),
    ],
)

# ─── Defaults ────────────────────────────────────────────────────────────────

BASE_MODEL   = "microsoft/deberta-v3-small"
MAX_LEN      = 512      # token budget — Args.me args average ~120 tokens
BATCH_SIZE   = 16
N_TRIALS     = 15      # Optuna HPO trials
PATIENCE     = 3        # early-stopping patience (epochs without val improvement)
MAX_EPOCHS   = 10

SEED         = 42

VAL_TOPICS  = [9, 20]   # Entertainment & Sports, World/International
TEST_TOPICS = [7, 19]   # Elections (Felon Voting), Sex & Gender (Born Gay?)

DEFAULT_HP = {
    "lr":            2e-5,
    "dropout":       0.1,
    "weight_decay":  0.01,
    "warmup_ratio":  0.1,
}


# ─── Dataset ──────────────────────────────────────────────────────────────────
def load_and_prepare(args_path: Path, topics_path: Path) -> tuple:
    """
    Load and preprocess Webis ArgQuality20.

    Returns:
        df         — DataFrame with columns: premise, query, score, topic_id
        score_min  — raw Combined Quality min (saved to config for inference)
        score_max  — raw Combined Quality max
    """
    args_df = pd.read_csv(args_path)
    topics_df = pd.read_csv(topics_path)

    # Filter to argumentative units only
    args_df = args_df[args_df["Is Argument?"] == True].copy()
    log.info("Loaded %d argumentative samples (filtered from %d total)",
             len(args_df), len(pd.read_csv(args_path))
             )

    # Join to get Long Query (central proposition) per topic
    df = args_df.merge(
        topics_df[["Topic ID", "Long Query"]],
        on="Topic ID", how="left",
        )
    if df["Long Query"].isnull().any():
        raise ValueError("Some arguments have no matching topic — check Topic IDs")

    # Normalise Combined Quality to [0, 1]
    score_min = float(df["Combined Quality"].min())
    score_max = float(df["Combined Quality"].max())
    df["score"] = (df["Combined Quality"] - score_min) / (score_max - score_min)

    df = df.rename(columns={"Premise": "premise", "Long Query": "query",
                            "Topic ID": "topic_id"
                            }
                   )
    log.info("Score range: raw [%.3f, %.3f] → normalised [%.3f, %.3f]",
             score_min, score_max, df["score"].min(), df["score"].max()
             )
    return df[["premise", "query", "score", "topic_id"]], score_min, score_max


def make_splits(df: pd.DataFrame) -> tuple:
    """
    Grouped split by topic_id.
    Val and test use held-out topic groups (unseen categories → generalisation test).
    """
    val_mask = df["topic_id"].isin(VAL_TOPICS)
    test_mask = df["topic_id"].isin(TEST_TOPICS)
    train_mask = ~val_mask & ~test_mask

    train = df[train_mask].reset_index(drop=True)
    val = df[val_mask].reset_index(drop=True)
    test = df[test_mask].reset_index(drop=True)

    log.info(
        "Split — train=%d (topics %s)  val=%d (topics %s)  test=%d (topics %s)",
        len(train), sorted(df[train_mask]["topic_id"].unique().tolist()),
        len(val), VAL_TOPICS,
        len(test), TEST_TOPICS,
        )
    return train, val, test


class ArgQualityDataset(Dataset):
    """
    Tokenises (query, premise) pairs as cross-encoder input:
        [CLS] query [SEP] premise [SEP]
    Returns input_ids, attention_mask, token_type_ids, and float label.
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int):
        self.encodings = tokenizer(
            df["query"].tolist(),
            df["premise"].tolist(),
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
            )
        self.labels = torch.tensor(df["score"].tolist(), dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


# ─── Model ────────────────────────────────────────────────────────────────────

class DeBERTaQualityModel(nn.Module):
    """
    Minimal cross-encoder regression head on DeBERTa-v3-small.
    [CLS] → Dropout → Linear(hidden_size, 1) → Sigmoid → ∈ [0, 1]
    """

    def __init__(self, base_model_name: str, dropout: float = 0.1):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(base_model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(hidden, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # DeBERTa-v3 does not use token_type_ids — drop silently if passed
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            )
        cls = self.dropout(out.last_hidden_state[:, 0, :])
        return self.sigmoid(self.regressor(cls)).squeeze(-1)  # (batch,)


# ─── Training helpers ─────────────────────────────────────────────────────────

def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _scheduler(optimizer, warmup_steps: int, total_steps: int):
    from transformers import get_linear_schedule_with_warmup
    return get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)


def train_one_epoch(model, loader, optimizer, scheduler, device) -> float:
    model.train()
    loss_fn    = nn.MSELoss()
    total_loss = 0.0
    for batch in loader:
        labels = batch.pop("labels").to(device)
        batch  = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        loss = loss_fn(model(**batch), labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    loss_fn = nn.MSELoss()
    preds, tgts = [], []
    total_loss = 0.0
    for batch in loader:
        labels = batch.pop("labels").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        p = model(**batch)
        total_loss += loss_fn(p, labels).item()
        preds.extend(p.cpu().numpy())
        tgts.extend(labels.cpu().numpy())

    preds_np = np.array(preds)
    tgts_np = np.array(tgts)
    r, _ = pearsonr(preds_np, tgts_np)
    return {
        "mse": round(total_loss / len(loader), 6),
        "mae": round(float(np.mean(np.abs(preds_np - tgts_np))), 6),
        "pearson_r": round(float(r), 6),
        }


def run_training(
        hp: dict, train_loader, val_loader, base_model_name: str,
        max_epochs: int, patience: int, device,
        save_path: Path = None
        ) -> tuple:
    """
    Train with given HP. Returns (best_val_mse, best_state_dict).
    Saves best checkpoint to save_path/pytorch_model.bin if provided.
    """
    model = DeBERTaQualityModel(base_model_name, dropout=hp["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=hp["lr"], weight_decay=hp["weight_decay"]
                                  )
    total_steps = len(train_loader) * max_epochs
    warmup_steps = int(total_steps * hp["warmup_ratio"])
    sched = _scheduler(optimizer, warmup_steps, total_steps)

    best_val_mse = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, sched, device)
        val_metrics = evaluate(model, val_loader, device)
        log.info(
            "  epoch %2d  train_loss=%.5f  val_mse=%.5f  val_mae=%.5f  val_r=%.4f",
            epoch, train_loss,
            val_metrics["mse"], val_metrics["mae"], val_metrics["pearson_r"],
            )
        if val_metrics["mse"] < best_val_mse:
            best_val_mse = val_metrics["mse"]
            best_state = deepcopy(model.state_dict())
            no_improve = 0
            if save_path:
                torch.save(best_state, save_path / "pytorch_model.bin")
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info("  Early stopping at epoch %d (patience=%d)", epoch, patience)
                break

    return best_val_mse, best_state


# ─── HPO ──────────────────────────────────────────────────────────────────────

def run_hpo(
        train_loader, val_loader, base_model_name: str,
        n_trials: int, max_epochs: int, patience: int, device
        ) -> dict:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        hp = {
            "lr": trial.suggest_float("lr", 1e-5, 5e-5, log=True),
            "dropout": trial.suggest_float("dropout", 0.05, 0.3),
            "weight_decay": trial.suggest_float("weight_decay", 1e-4, 0.1, log=True),
            "warmup_ratio": trial.suggest_float("warmup_ratio", 0.05, 0.2),
            }
        log.info("Trial %d  hp=%s", trial.number, hp)
        val_mse, _ = run_training(hp, train_loader, val_loader,
                                  base_model_name, max_epochs, patience, device
                                  )
        return val_mse

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
        )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    log.info("HPO complete — best val MSE=%.5f  best HP=%s",
             study.best_value, study.best_params
             )
    return study.best_params


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    import os
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    parser = argparse.ArgumentParser(
        description="Fine-tune DeBERTa-v3-small for pairwise argument quality regression"
    )
    parser.add_argument("--args-data",   "-a", required=True,
                        help="Path to webis-argquality20-full.csv")
    parser.add_argument("--topic-data",  "-t", required=True,
                        help="Path to webis-argquality20-topics.csv")
    parser.add_argument("--output",      "-o", default="./quality_model",
                        help="Directory to save model artefacts")
    parser.add_argument("--base-model",  default=BASE_MODEL)
    parser.add_argument("--max-len",     type=int, default=MAX_LEN)
    parser.add_argument("--batch-size",  type=int, default=BATCH_SIZE)
    parser.add_argument("--max-epochs",  type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience",    type=int, default=PATIENCE)
    parser.add_argument("--n-trials",    type=int, default=N_TRIALS)
    parser.add_argument("--no-hpo",      action="store_true",
                        help="Skip HPO and use default hyperparameters")
    parser.add_argument("--eval-only",   action="store_true",
                        help="Skip training; load saved checkpoint and run test evaluation")
    parser.add_argument("--verbose",     "-v", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device     = _device()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Device: %s  base_model: %s", device, args.base_model)
    # ── Tokenizer ─────────────────────────────────────────────────────────────
    from transformers import AutoTokenizer
    log.info("Loading tokenizer: %s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.save_pretrained(output_dir / "tokenizer")

    # ── Data ──────────────────────────────────────────────────────────────────
    df, score_min, score_max = load_and_prepare(
        Path(args.args_data), Path(args.topic_data)
        )
    train_df, val_df, test_df = make_splits(df)

    train_ds = ArgQualityDataset(train_df, tokenizer, args.max_len)
    val_ds = ArgQualityDataset(val_df, tokenizer, args.max_len)
    test_ds = ArgQualityDataset(test_df, tokenizer, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True
                              )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True
                            )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=0, pin_memory=True
                             )

    # ── Eval-only mode ────────────────────────────────────────────────────────
    if args.eval_only:
        log.info("Eval-only — loading checkpoint from %s", output_dir)
        cfg = json.loads((output_dir / "config.json").read_text())
        model = DeBERTaQualityModel(
            cfg["base_model"], dropout=cfg["hyperparams"]["dropout"]
            ).to(device)
        model.load_state_dict(
            torch.load(output_dir / "pytorch_model.bin", map_location=device)
            )
        metrics = evaluate(model, test_loader, device)
        log.info("Test — MSE=%.5f  MAE=%.5f  Pearson r=%.4f",
                 metrics["mse"], metrics["mae"], metrics["pearson_r"]
                 )
        (output_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
        return

    # ── HPO ───────────────────────────────────────────────────────────────────
    if args.no_hpo:
        best_hp = DEFAULT_HP
        log.info("Skipping HPO — using default HP: %s", best_hp)
    else:
        log.info("Starting HPO — %d trials", args.n_trials)
        best_hp = {**DEFAULT_HP, **run_hpo(
            train_loader, val_loader, args.base_model,
            args.n_trials, args.max_epochs, args.patience, device,
            )
                   }

    # ── Final training on train + val with best HP ─────────────────────────────
    # Combine train and val for final model
    final_df = pd.concat([train_df, val_df], ignore_index=True)
    final_ds = ArgQualityDataset(final_df, tokenizer, args.max_len)
    final_loader = DataLoader(final_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True
                              )

    log.info("Final training on train+val (%d samples) with HP: %s",
             len(final_df), best_hp
             )
    _, best_state = run_training(
        best_hp, final_loader, val_loader,
        args.base_model, args.max_epochs, args.patience,
        device, save_path=output_dir,
        )

    # ── Save config ────────────────────────────────────────────────────────────
    config = {
        "base_model": args.base_model,
        "max_len": args.max_len,
        "score_min": score_min,
        "score_max": score_max,
        "val_topics": VAL_TOPICS,
        "test_topics": TEST_TOPICS,
        "hyperparams": best_hp,
        }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    log.info("Config saved → %s/config.json", output_dir)

    # ── Test evaluation ────────────────────────────────────────────────────────
    model = DeBERTaQualityModel(args.base_model, dropout=best_hp["dropout"]).to(device)
    model.load_state_dict(
        torch.load(output_dir / "pytorch_model.bin", map_location=device)
        )
    test_metrics = evaluate(model, test_loader, device)
    log.info(
        "Test (unseen topics %s) — MSE=%.5f  MAE=%.5f  Pearson r=%.4f",
        TEST_TOPICS,
        test_metrics["mse"], test_metrics["mae"], test_metrics["pearson_r"],
        )
    (output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    log.info("All artefacts saved → %s", output_dir)


if __name__ == "__main__":
    main()
