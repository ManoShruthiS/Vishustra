"""Datasets package — tokenization, tasks, batching, splitting (plan.md §35)."""

from app.datasets.loader import make_loaders, seed_all
from app.datasets.preprocessing import collate_batch
from app.datasets.splitter import SplitView, train_val_split
from app.datasets.task import CopyTask
from app.datasets.tokenizer import CharacterTokenizer

__all__ = [
    "CharacterTokenizer",
    "CopyTask",
    "SplitView",
    "collate_batch",
    "make_loaders",
    "seed_all",
    "train_val_split",
]