"""PHASE 5 — tokenizer tests (plan.md §65 'Tokenizer')."""

from app.datasets.tokenizer import EOS, PAD, SOS, CharacterTokenizer


def test_special_ids_are_reserved_first():
    tok = CharacterTokenizer(chars="ab")
    assert tok.id_to_token[0] == PAD
    assert tok.id_to_token[1] == SOS
    assert tok.id_to_token[2] == EOS
    assert tok.token_to_id["a"] == 3
    assert tok.token_to_id["b"] == 4
    assert tok.vocab_size == 5


def test_encode_decode_roundtrip():
    tok = CharacterTokenizer(chars="abc")
    assert tok.encode("cab") == [5, 3, 4]
    assert "".join(tok.decode(tok.encode("abcabc"))) == "abcabc"


def test_from_text_scans_unique_chars():
    tok = CharacterTokenizer.from_text("hello world")
    assert tok.encode("h")[0] == tok.encode("h")[0]
    assert "".join(tok.decode(tok.encode("helo wrd"))) == "helo wrd"


def test_encode_with_ends_wraps_sos_eos():
    tok = CharacterTokenizer(chars="a")
    ids = tok.encode_with_ends("a")
    assert ids == [tok.sos_id, tok.encode("a")[0], tok.eos_id]


def test_duplicate_chars_are_deduplicated():
    tok = CharacterTokenizer(chars="aabbbc")
    assert tok.vocab_size == 3 + 3  # PAD, SOS, EOS + {a,b,c}