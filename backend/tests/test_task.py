"""PHASE 5 — CopyTask dataset + splitter tests."""

from app.datasets import CopyTask, train_val_split


def test_copytask_roundtrip_lengths():
    task = CopyTask(vocab_size=8, n_examples=100, min_len=2, max_len=6, seed=0)
    assert len(task) == 100
    for src, tgt in [task[i] for i in (0, 7, 42, 99)]:
        assert len(src) == len(tgt)
        assert src == tgt


def test_copytask_deterministic():
    a = CopyTask(vocab_size=8, n_examples=20, seed=3)
    b = CopyTask(vocab_size=8, n_examples=20, seed=3)
    assert a.pairs == b.pairs


def test_copytask_reverse_mode():
    task = CopyTask(vocab_size=8, n_examples=20, seed=0, reverse=True)
    for src, tgt in task.pairs:
        assert tgt == list(reversed(src))


def test_copytask_ids_within_vocab():
    task = CopyTask(vocab_size=8, n_examples=200, seed=0)
    for src, tgt in task.pairs:
        assert all(0 <= i < 8 for i in src)
        assert all(0 <= i < 8 for i in tgt)


def test_train_val_split_partitions():
    task = CopyTask(vocab_size=8, n_examples=100, seed=0)
    train, val = train_val_split(task, val_fraction=0.2, seed=7)
    assert len(train) == 80
    assert len(val) == 20


def test_split_views_look_through_to_items():
    task = CopyTask(vocab_size=8, n_examples=10, seed=0)
    _, val = train_val_split(task, val_fraction=0.5, seed=0)
    assert val[0][0] in (p[0] for p in task.pairs)