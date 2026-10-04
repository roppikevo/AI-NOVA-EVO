"""Tests for the NOVA tokenizer and text corpus builder (offline)."""

import json

import pytest

pytest.importorskip("tokenizers")

from evo.corpus.build_text_v1 import build, split_of
from evo.corpus.sources import Document, clean_wiki_text
from nova.data import TokenSequenceDataset
from nova.tokenizer import EOS_ID, PAD_ID, NovaTokenizer, pack_sequences

SAMPLES = {
    "sk": "Bratislava je hlavné mesto Slovenska. Leží na brehu Dunaja, "
          "pri hraniciach s Rakúskom a Maďarskom. Žije tu približne "
          "štyristotisíc obyvateľov.",
    "cs": "Praha je hlavní a zároveň největší město Česka. Leží na řece "
          "Vltavě a žije v ní přes milion obyvatel.",
    "pl": "Warszawa jest stolicą Polski. Leży nad Wisłą, a jej historia "
          "sięga średniowiecza. Zażółć gęślą jaźń.",
    "en": "London is the capital of the United Kingdom. It stands on the "
          "River Thames and has a population of about nine million.",
    "py": "def fibonacci(n: int) -> int:\n    a, b = 0, 1\n    for _ in "
          "range(n):\n        a, b = b, a + b\n    return a\n",
    "rs": "fn main() {\n    let v: Vec<i32> = (1..=10).collect();\n    "
          "let s: i32 = v.iter().sum();\n    println!(\"{}\", s);\n}\n",
}


def _groups(copies=60):
    groups = {}
    for lang, text in SAMPLES.items():
        groups[lang] = [
            Document(lang, "python-stdlib" if lang == "py" else
                     "rust-src" if lang == "rs" else "wikipedia",
                     f"{text}\n# {i}" if lang in ("py", "rs") else f"{text} ({i})")
            for i in range(copies)
        ]
    return groups


@pytest.fixture(scope="module")
def tok():
    texts = [d.text for docs in _groups().values() for d in docs]
    return NovaTokenizer.train(texts, vocab_size=600, min_frequency=1)


def test_special_token_ids_fixed(tok):
    assert tok.encode_document("x", "sk")[0] == 4
    assert tok.lang_id("rs") == 9
    assert PAD_ID == 0 and EOS_ID == 3


@pytest.mark.parametrize("lang", list(SAMPLES))
def test_roundtrip_all_languages(tok, lang):
    text = SAMPLES[lang]
    assert tok.decode(tok.encode(text)) == text


def test_unseen_characters_still_roundtrip(tok):
    text = "Ďakujem! Ľúbim ťa. Łódź. 🙂 λ → ∑"
    assert tok.decode(tok.encode(text)) == text


def test_pack_sequences_exact_length_and_order():
    docs = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    out = list(pack_sequences(docs, 4))
    assert out == [[1, 2, 3, 4], [5, 6, 7, 8]]


def test_split_is_deterministic():
    d = Document("sk", "wikipedia", "rovnaký text")
    assert split_of(d) == split_of(Document("sk", "wikipedia", "rovnaký text"))


def test_clean_wiki_cuts_reference_tail():
    text = "Úvod článku.\n\nHistória\nText.\n\nReferencie\n1. zdroj"
    assert clean_wiki_text(text) == "Úvod článku.\n\nHistória\nText."


def test_build_produces_training_compatible_dataset(tmp_path):
    manifest = build(_groups(), tmp_path, seq_len=32, vocab_size=600,
                     log=lambda *_: None)

    for split in ("train", "val", "test"):
        assert (tmp_path / f"{split}.txt").exists()
    assert manifest["splits"]["train"]["sequences"] > 0

    ds = TokenSequenceDataset(tmp_path / "train.txt", seq_len=32)
    x, y = ds[0]
    assert x.shape[0] == 31 and y.shape[0] == 31
    assert int(ds.samples[0][0]) >= 0

    tok = NovaTokenizer.load(tmp_path / "tokenizer.json")
    assert max(max(s) for s in ds.samples) < tok.vocab_size

    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["sources"]["sk"]["license"] == "CC BY-SA 4.0"
    assert set(m["bytes_per_token"]) <= set(SAMPLES)


def test_no_document_in_two_splits(tmp_path):
    groups = _groups()
    seen = {}
    for docs in groups.values():
        for d in docs:
            s = split_of(d)
            assert seen.setdefault(d.key, s) == s


def test_nbsp_normalized():
    assert clean_wiki_text("Bratislava je v strede mesta") == "Bratislava je v strede mesta"


def test_identity_documents_mention_creator():
    from evo.corpus.identity import CREATOR, identity_documents

    docs = list(identity_documents(300))
    assert len(docs) == 300
    assert all(CREATOR in d.text for d in docs)
    assert {d.lang for d in docs} == {"sk", "cs", "pl", "en"}
    assert len({d.key for d in docs}) == 300
    assert not any("heslo" in d.text.lower() or "password" in d.text.lower() for d in docs)
