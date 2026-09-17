"""PHASE 5 — end-to-end trainer smoke test.

Trains a tiny Transformer on the copy task for a few epochs and checks:
loss decreases, history is recorded, checkpoints round-trip through
save/load, and the optimizer carries the paper's Adam hyperparameters.
"""

import torch
from app.core.config import PAPER_TRAINING, PaperConfig
from app.datasets import CharacterTokenizer, CopyTask, make_loaders
from app.training import (
    TrainConfig,
    Trainer,
    greedy_decode,
    load_checkpoint,
    paper_adam,
    save_checkpoint,
)
from app.transformer import Transformer


def tiny_copy_config() -> PaperConfig:
    return PaperConfig(
        name="tiny-copy-test",
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_model=8,
        d_ff=16,
        num_heads=4,
        d_k=2,
        d_v=2,
        dropout=0.0,
        label_smoothing=0.1,
        train_steps=0,
        params_millions=0,
        notes=("Phase 5 trainer smoke test.",),
    )


def test_paper_adam_hyperparameters():
    tok = CharacterTokenizer(chars="a")
    model = Transformer(tiny_copy_config(), tok.vocab_size, tok.vocab_size, max_len=6)
    opt = paper_adam(model, lr=0.5)
    group = opt.param_groups[0]
    assert group["betas"] == (PAPER_TRAINING.beta1, PAPER_TRAINING.beta2)
    assert group["eps"] == PAPER_TRAINING.epsilon


def test_training_decreases_loss_and_records_history(tmp_path):
    tok = CharacterTokenizer(chars="abcd")
    vocab = tok.vocab_size
    train_ds = CopyTask(vocab_size=vocab, n_examples=48, min_len=2, max_len=5, seed=1, content_start=3)
    val_ds = CopyTask(vocab_size=vocab, n_examples=12, min_len=2, max_len=5, seed=2, content_start=3)
    train_loader, val_loader = make_loaders(train_ds, val_ds, tok, batch_size=8, max_len=6)
    model = Transformer(tiny_copy_config(), vocab, vocab, max_len=6)
    trainer = Trainer(
        model,
        TrainConfig(epochs=3, batch_size=8, init_lr=0.5, warmup_steps=15, max_len=6,
                    checkpoint_dir=str(tmp_path), run_name="smoke"),
        vocab_size=vocab,
    )
    history = trainer.fit(train_loader, val_loader)
    assert len(history) == 3
    assert history[0]["epoch"] == 1.0
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    assert 0.0 <= history[-1]["val_acc"] <= 1.0
    assert history[-1]["val_ppl"] > 1.0
    assert history[-1]["lr"] > 0.0
    assert (tmp_path / "smoke" / "best.pt").exists()
    assert (tmp_path / "smoke" / "epoch_003.pt").exists()


def test_checkpoint_round_trip(tmp_path):
    tok = CharacterTokenizer(chars="abcd")
    vocab = tok.vocab_size
    model = Transformer(tiny_copy_config(), vocab, vocab, max_len=6)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    checkpoint_path = save_checkpoint(
        tmp_path / "ck.pt", model, optimizer=None, scheduler=None, epoch=2, history=[{"x": 1.0}],
        vocab=tok.tokens,
    )
    state = load_checkpoint(checkpoint_path, model)
    assert state["epoch"] == 2
    assert state["history"] == [{"x": 1.0}]
    assert state["config"].name == "tiny-copy-test"
    assert state["vocab"] == tok.tokens
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k])


def test_greedy_decode_runs_on_trained_model(tmp_path):
    tok = CharacterTokenizer(chars="abcd")
    vocab = tok.vocab_size
    train_ds = CopyTask(vocab_size=vocab, n_examples=48, min_len=2, max_len=4, seed=5, content_start=3)
    val_ds = CopyTask(vocab_size=vocab, n_examples=8, min_len=2, max_len=4, seed=6, content_start=3)
    train_loader, val_loader = make_loaders(train_ds, val_ds, tok, batch_size=8, max_len=5)
    model = Transformer(tiny_copy_config(), vocab, vocab, max_len=5)
    trainer = Trainer(
        model,
        TrainConfig(epochs=4, batch_size=8, init_lr=0.5, warmup_steps=15, max_len=5,
                    checkpoint_dir=str(tmp_path), run_name="smoke"),
        vocab_size=vocab,
    )
    trainer.fit(train_loader, val_loader)
    src, _ = val_ds.tensor_pairs()
    ids = src[0][src[0] != 0].tolist()
    out = greedy_decode(model, torch.tensor(ids), tok, max_steps=8)
    assert all(0 <= i < vocab for i in out)
    assert len(out) <= 8