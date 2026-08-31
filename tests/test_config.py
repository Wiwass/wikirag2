"""Config tests: KB root resolution, lifecycle, sliders, per-KB config."""

from __future__ import annotations

from pathlib import Path

import pytest

from wikirag.config import (
    DB_FILENAME,
    DEFAULT_KB_CONFIG,
    KB_SUBDIRS,
    KnowledgeBase,
    RetrievalDepth,
    SynthesisLevel,
    create_kb,
    kb_root,
    list_kbs,
    open_kb,
)


@pytest.fixture()
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("WIKIRAG_HOME", str(tmp_path / "kbs"))
    return tmp_path / "kbs"


# ------------------------------------------------------------------ sliders


def test_slider_enums() -> None:
    assert [level.value for level in SynthesisLevel] == ["S0", "S1", "S2", "S3"]
    assert [depth.value for depth in RetrievalDepth] == ["R0", "R1", "R2", "R3"]
    assert SynthesisLevel.parse("s2") is SynthesisLevel.S2
    assert SynthesisLevel.parse(SynthesisLevel.S1) is SynthesisLevel.S1
    assert RetrievalDepth.parse(" r3 ") is RetrievalDepth.R3
    assert SynthesisLevel.S1 < SynthesisLevel.S3
    assert RetrievalDepth.R2 >= RetrievalDepth.R1
    assert SynthesisLevel.S2 == "S2"  # StrEnum: usable directly as a config value
    with pytest.raises(ValueError):
        SynthesisLevel.parse("S9")
    with pytest.raises(ValueError):
        RetrievalDepth.parse("deep")


# --------------------------------------------------------------------- root


def test_kb_root_env_and_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WIKIRAG_HOME", raising=False)
    monkeypatch.chdir(tmp_path)
    assert kb_root() == tmp_path / "knowledge_bases"
    monkeypatch.setenv("WIKIRAG_HOME", str(tmp_path / "elsewhere"))
    assert kb_root() == tmp_path / "elsewhere"


def test_list_kbs_on_missing_root(home: Path) -> None:
    assert list_kbs() == []


# ---------------------------------------------------------------- lifecycle


def test_create_open_list(home: Path) -> None:
    with create_kb("arsmagica") as kb:
        assert isinstance(kb, KnowledgeBase)
        assert kb.name == "arsmagica"
        assert kb.root == home / "arsmagica"
        assert kb.db_path.is_file()
        for sub in KB_SUBDIRS:
            assert (kb.root / sub).is_dir()
        assert kb.vault_dir.name == "vault" and kb.raw_dir.name == "raw"
        assert kb.archive_dir.name == "archive" and kb.vectors_dir.name == "vectors"
        # the store is open and migrated
        kb.store.add_document(doc_id="d1", filename="d1.pdf")

    create_kb("second").close()
    assert list_kbs() == ["arsmagica", "second"]
    # a directory without kb.sqlite is not a KB
    (home / "notakb").mkdir()
    assert list_kbs() == ["arsmagica", "second"]

    with open_kb("arsmagica") as kb2:
        assert kb2.store.get_document("d1")["filename"] == "d1.pdf"


def test_create_twice_and_open_missing(home: Path) -> None:
    create_kb("dup").close()
    with pytest.raises(FileExistsError):
        create_kb("dup")
    with pytest.raises(FileNotFoundError):
        open_kb("ghost")


def test_invalid_names_rejected(home: Path) -> None:
    for name in ["", "../escape", "with/slash", ".hidden", "a b"]:
        with pytest.raises(ValueError):
            create_kb(name)


def test_explicit_root_overrides_env(home: Path, tmp_path: Path) -> None:
    other = tmp_path / "other_root"
    create_kb("kb", root=other).close()
    assert (other / "kb" / DB_FILENAME).is_file()
    assert list_kbs(root=other) == ["kb"]
    assert list_kbs() == []
    open_kb("kb", root=other).close()


# ------------------------------------------------------------------- config


def test_defaults_and_config_roundtrip(home: Path) -> None:
    with create_kb("cfg") as kb:
        assert kb.config == DEFAULT_KB_CONFIG
        assert kb.synthesis_level is SynthesisLevel.S0
        assert kb.default_depth is RetrievalDepth.R1

        kb.synthesis_level = SynthesisLevel.S2
        kb.set_config("embedding_model", "bge-m3")
        assert kb.synthesis_level is SynthesisLevel.S2
        assert kb.config["embedding_model"] == "bge-m3"

    with open_kb("cfg") as kb2:
        assert kb2.synthesis_level is SynthesisLevel.S2
        assert kb2.config["embedding_model"] == "bge-m3"
        kb2.synthesis_level = "s3"  # strings accepted, normalized
        assert kb2.store.config_get("synthesis_level") == "S3"


def test_create_kb_accepts_config_overrides(home: Path) -> None:
    with create_kb("preset", config={"synthesis_level": "S2", "role.enrich": "flash"}) as kb:
        assert kb.synthesis_level is SynthesisLevel.S2
        assert kb.config["role.enrich"] == "flash"
        assert kb.default_depth is RetrievalDepth.R1  # untouched default kept
