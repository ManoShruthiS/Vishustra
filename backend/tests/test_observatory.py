"""PHASE 6 — Attention Observatory end-to-end smoke test (plan.md §8).

Builds a tiny untrained Transformer, runs the full ``observe`` pipeline on
one copy example and asserts the whole visualization family is written to
disk.  Untrained weights are fine here — this validates the *pipeline*,
not the attention quality.
"""


from app.core.config import PaperConfig
from app.datasets import CharacterTokenizer, CopyTask
from app.transformer import Transformer
from app.visualization.demo_observatory import observe

EXPECTED = {
    "attention_matrix.png",
    "head_grid_encoder_l0.png",
    "head_grid_cross_l0.png",
    "layer_strip_encoder.png",
    "layer_strip_cross.png",
    "layer_strip_decoder_self.png",
    "attention_graph.png",
    "attention_graph_cross.png",
    "positional_encoding.png",
}


def tiny_config() -> PaperConfig:
    return PaperConfig(
        name="tiny-observatory",
        num_encoder_layers=1,
        num_decoder_layers=1,
        d_model=8,
        d_ff=16,
        num_heads=2,
        d_k=4,
        d_v=4,
        dropout=0.0,
        label_smoothing=0.0,
        train_steps=0,
        params_millions=0,
        notes=("Phase 6 observatory smoke test.",),
    )


def test_observe_produces_full_visual_family(tmp_path):
    tok = CharacterTokenizer(chars="abcd")
    model = Transformer(tiny_config(), tok.vocab_size, tok.vocab_size, max_len=8)
    task = CopyTask(vocab_size=tok.vocab_size, n_examples=4, min_len=2, max_len=4, seed=3, content_start=3)
    src, tgt = task.tensor_pairs()
    src_ids = src[0][src[0] != 0].tolist()
    tgt_ids = tgt[0][tgt[0] != 0].tolist()

    artifacts = observe(model, tok, src_ids, tgt_ids, tmp_path)
    names = {p.name for p in artifacts}
    assert EXPECTED <= names
    for p in artifacts:
        assert p.stat().st_size > 0, p
    assert (tmp_path / "head_grid_encoder_l0.png").exists()


def test_observe_history_renders_training_curve(tmp_path):
    tok = CharacterTokenizer(chars="abcd")
    model = Transformer(tiny_config(), tok.vocab_size, tok.vocab_size, max_len=8)
    task = CopyTask(vocab_size=tok.vocab_size, n_examples=2, min_len=2, max_len=3, seed=5, content_start=3)
    src, tgt = task.tensor_pairs()
    src_ids = src[0][src[0] != 0].tolist()
    tgt_ids = tgt[0][tgt[0] != 0].tolist()
    artifacts = observe(
        model, tok, src_ids, tgt_ids, tmp_path,
        history=[{"epoch": 1.0, "train_loss": 3.0, "val_loss": 3.1, "train_acc": 0.1, "val_acc": 0.1}],
    )
    assert (tmp_path / "training_curve.png") in artifacts