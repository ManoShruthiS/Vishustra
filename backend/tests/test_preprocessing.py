"""PHASE 5 — collate / batching tests.

Verifies teacher-forced shape conventions: tgt_in = [SOS] + tokens,
tgt_out = tokens + [EOS], variable lengths padded, and length tensors
that drive the causal + padding masks in the trainer.
"""

import torch
from app.datasets import CharacterTokenizer, CopyTask
from app.datasets.preprocessing import collate_batch
from app.training.evaluator import src_mask_for, tgt_mask_for


def _batch():
    tok = CharacterTokenizer(chars="abc")
    task = CopyTask(vocab_size=tok.vocab_size, n_examples=6, min_len=2, max_len=4, seed=0)
    return collate_batch([task[i] for i in range(6)], tok, max_len=6), tok, task


def test_teacher_forcing_alignment():
    batch, tok, task = _batch()
    for i in range(len(task)):
        _, tgt = task[i]
        tgt_in = batch["tgt_in"][i]
        tgt_out = batch["tgt_out"][i]
        valid = batch["tgt_len"][i]
        assert torch.equal(tgt_out[: len(tgt)], torch.tensor(tgt))
        assert int(tgt_out[len(tgt)]) == tok.eos_id
        assert int(tgt_in[0]) == tok.sos_id
        assert int(valid) == len(tgt) + 1


def test_padding_positions_are_pad_id():
    batch, tok, _ = _batch()
    for i in range(batch["tgt_in"].shape[0]):
        valid = int(batch["tgt_len"][i])
        assert torch.all(batch["tgt_in"][i, valid:] == tok.pad_id)
        assert torch.all(batch["tgt_out"][i, valid:] == tok.pad_id)


def test_src_tgt_length_tensors():
    batch, _, task = _batch()
    for i in range(len(task)):
        src, _ = task[i]
        assert int(batch["src_len"][i]) == len(src)
        assert int(batch["tgt_len"][i]) == len(src) + 1


def test_masks_broadcast_to_batch_dimensions():
    batch, _, _ = _batch()
    src_len = batch["src"].shape[1]
    tgt_len = batch["tgt_in"].shape[1]
    s_mask = src_mask_for(batch, src_len)
    t_mask = tgt_mask_for(batch, tgt_len)
    assert s_mask.shape == (batch["src"].shape[0], 1, src_len)
    assert t_mask.shape == (batch["tgt_in"].shape[0], tgt_len, tgt_len)
    mb = t_mask == float("-inf")
    n = batch["src"].shape[0]
    for i in range(n):
        row_valid = int(batch["tgt_len"][i])
        assert torch.all(mb[i, : row_valid, row_valid:])


def test_single_example_collate():
    tok = CharacterTokenizer(chars="ab")
    task = CopyTask(vocab_size=tok.vocab_size, n_examples=1, seed=4)
    batch = collate_batch([task[0]], tok)
    assert batch["src"].shape[0] == 1