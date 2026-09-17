"""PHASE 5 — Character tokenizer.

Tokenizes text into integer ids with special tokens PAD / SOS / EOS
reserved at the front of the vocabulary (plan.md §65 'Tokenizer',
LAB 02). The toy seq2seq tasks operate over a small alphabet so a full
subword tokenizer is not needed at this phase.
"""

from __future__ import annotations

PAD = "<PAD>"
SOS = "<SOS>"
EOS = "<EOS>"

_SPECIAL = (PAD, SOS, EOS)


class CharacterTokenizer:
    def __init__(self, chars: str | None = None, special: tuple[str, ...] = _SPECIAL) -> None:
        tokens = list(special)
        if chars is not None:
            for ch in chars:
                if ch not in tokens:
                    tokens.append(ch)
        self.tokens = tokens
        self.id_to_token = {i: t for i, t in enumerate(tokens)}
        self.token_to_id = {t: i for i, t in enumerate(tokens)}

    @classmethod
    def from_text(cls, text: str, special: tuple[str, ...] = _SPECIAL) -> CharacterTokenizer:
        chars = ""
        for ch in text:
            if ch not in chars:
                chars += ch
        return cls(chars=chars, special=special)

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD]

    @property
    def sos_id(self) -> int:
        return self.token_to_id[SOS]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[EOS]

    def encode(self, seq: str | list[str]) -> list[int]:
        return [self.token_to_id[ch] for ch in seq]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.id_to_token[int(i)] for i in ids]

    def encode_with_ends(self, seq: str | list[str]) -> list[int]:
        """``[SOS] + ids + [EOS]`` for a tokenizer.sentence -> tokens line."""
        return [self.sos_id, *self.encode(seq), self.eos_id]