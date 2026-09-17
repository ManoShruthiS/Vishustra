"""PHASE 7 — Experiment runner (plan.md §67 'Execution, Logging, Metrics').

Turns a registered experiment into a completed run: dataset + model
construction, training, held-out evaluation, attention statistics and a
persisted *reproducibility package* (plan.md §31) plus metrics/history
JSON inside the experiment directory.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from app.core.config import PAPER_TRAINING
from app.datasets import CharacterTokenizer, CopyTask, make_loaders, seed_all
from app.experiments.config import ExperimentConfig
from app.experiments.registry import ExperimentRecord, ExperimentRegistry
from app.training import TrainConfig, Trainer
from app.training.evaluator import src_mask_for, tgt_mask_for
from app.transformer import Transformer

logger = logging.getLogger("vishustra.experiments")


@dataclass
class ExperimentResult:
    """Everything one finished run produced and measured."""

    exp_id: str
    config: ExperimentConfig
    history: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    reproducibility: dict = field(default_factory=dict)
    artifacts: list[Path] = field(default_factory=list)


def software_environment() -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
    }


def dataset_version(config: ExperimentConfig) -> str:
    """Reproducible digest of the dataset + tokenizer definition (§31)."""
    blob = json.dumps(config.dataset, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


@torch.no_grad()
def attention_statistics(model: torch.nn.Module, loader, num_batches: int = 2) -> dict:
    """Attention entropy and concentration (plan.md §71 attention metrics).

    Averaged over every head and every encoder layer / decoder cross
    attention, over the first ``num_batches`` of a validation loader.
    """
    model.eval()
    entropies: list[float] = []
    concentrations: list[float] = []
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        tgt_len = batch["tgt_in"].shape[1]
        src_len = batch["src"].shape[1]
        _, _, trace = model(
            batch["src"],
            batch["tgt_in"],
            src_mask=src_mask_for(batch, src_len),
            tgt_mask=tgt_mask_for(batch, tgt_len),
        )
        groups = [trace.encoder.layers, trace.decoder.layers]
        for layers, kind in zip(groups, ("attention", "cross_attention")):
            for layer in layers:
                mha = layer.attention if kind == "attention" else layer.cross_attention
                for head in mha.heads:
                    w = head.weights[0].clamp_min(1e-12)
                    entropies.append(-(w * w.log()).sum(-1).mean().item())
                    concentrations.append(head.weights[0].max(-1).values.mean().item())
    if not entropies:
        return {"attention_entropy": float("nan"), "attention_concentration": float("nan")}
    n = len(entropies)
    return {
        "attention_entropy": sum(entropies) / n,
        "attention_concentration": sum(concentrations) / n,
    }


def run_experiment(
    registry: ExperimentRegistry,
    exp_id: str,
    *,
    save_model: bool = True,
) -> ExperimentResult:
    """Execute the registered experiment end to end and persist everything."""
    record: ExperimentRecord = registry.get(exp_id)
    if record.status == "running":
        raise RuntimeError(f"{exp_id} is already running")
    cfg = record.config_obj
    exp_dir = registry.root / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    registry.update(exp_id, status="running")

    file_handler = logging.FileHandler(exp_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)

    seed_all(cfg.seed)
    tokenizer = CharacterTokenizer(chars=cfg.dataset["alphabet"])
    vocab = tokenizer.vocab_size

    train_ds = CopyTask(
        vocab_size=vocab,
        n_examples=int(cfg.dataset["n_examples"]),
        min_len=int(cfg.dataset["min_len"]),
        max_len=int(cfg.dataset["max_len"]),
        seed=cfg.seed,
        content_start=int(cfg.dataset["content_start"]),
    )
    val_ds = CopyTask(
        vocab_size=vocab,
        n_examples=int(cfg.dataset["val_n_examples"]),
        min_len=int(cfg.dataset["min_len"]),
        max_len=int(cfg.dataset["max_len"]),
        seed=cfg.seed + 1,
        content_start=int(cfg.dataset["content_start"]),
    )
    train_loader, val_loader = make_loaders(
        train_ds, val_ds, tokenizer,
        batch_size=int(cfg.training["batch_size"]),
        max_len=int(cfg.architecture.get("max_len") or 8),
    )

    paper = cfg.as_paper_config()
    max_len = int(cfg.architecture.get("max_len") or 8)
    model = Transformer(paper, src_vocab_size=vocab, tgt_vocab_size=vocab, max_len=max_len)
    trainer = Trainer(
        model,
        TrainConfig(
            epochs=int(cfg.training["epochs"]),
            batch_size=int(cfg.training["batch_size"]),
            init_lr=float(cfg.training["init_lr"]),
            warmup_steps=int(cfg.training["warmup_steps"]),
            max_len=max_len,
            checkpoint_dir=str(exp_dir / "checkpoints"),
            run_name="run",
            seed=cfg.seed,
        ),
        vocab_size=vocab,
    )

    started = time.perf_counter()
    try:
        history = trainer.fit(train_loader, val_loader)
    except Exception:
        registry.update(exp_id, status="failed")
        raise
    finally:
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()
    train_time_s = time.perf_counter() - started

    stats = attention_statistics(model, val_loader)
    params = model.parameters_count()
    last = history[-1]

    metrics = {
        "epochs": float(int(cfg.training["epochs"])),
        "train_loss": last["train_loss"],
        "train_acc": last["train_acc"],
        "val_loss": last["val_loss"],
        "val_acc": last["val_acc"],
        "val_ppl": last["val_ppl"],
        "final_lr": last["lr"],
        "params": params,
        "params_millions": round(params / 1e6, 6),
        "train_time_s": round(train_time_s, 2),
        "training_steps": len(train_loader) * int(cfg.training["epochs"]),
        **stats,
    }
    repro = {
        "experiment_id": exp_id,
        "random_seed": cfg.seed,
        "dataset_version": dataset_version(cfg),
        "dataset": cfg.dataset,
        "model_configuration": paper.__dict__.copy(),
        "hyperparameters": {
            **cfg.training,
            "optimizer": PAPER_TRAINING.optimizer,
            "betas": (PAPER_TRAINING.beta1, PAPER_TRAINING.beta2),
            "epsilon": PAPER_TRAINING.epsilon,
            "scheduler": "Noam warmup (eq. 3)",
            "label_smoothing": paper.label_smoothing,
        },
        "software_environment": software_environment(),
        "training_steps": metrics["training_steps"],
        "metrics": {k: metrics[k] for k in ("val_loss", "val_acc", "val_ppl", "train_time_s")},
    }

    artifacts: list[Path] = []
    writers = {
        "metrics.json": lambda: json.dumps(metrics, indent=2),
        "history.json": lambda: json.dumps(history, indent=2),
        "reproducibility.json": lambda: json.dumps(repro, indent=2),
    }
    for filename, to_text in writers.items():
        target = exp_dir / filename
        target.write_text(to_text(), encoding="utf-8")
        artifacts.append(target)
    if save_model:
        model_path = exp_dir / "model.pt"
        torch.save(model.state_dict(), model_path)
        artifacts.append(model_path)

    registry.update(exp_id, status="completed", result=metrics)
    return ExperimentResult(
        exp_id=exp_id,
        config=cfg,
        history=history,
        metrics=metrics,
        reproducibility=repro,
        artifacts=artifacts,
    )