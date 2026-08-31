"""Configuration: KB registry and the two sliders.

Responsibilities
----------------
- Locate the knowledge-bases root and enumerate/create/open KBs by name.
- Define the `KnowledgeBase` handle: name, paths, open store, config. It is
  constructed in exactly one place (here) and passed to every operation —
  no function signatures accumulating loose paths and dicts (v1 lesson).
- Per-KB config lives in the store's `kb_config` table, not in module
  constants: synthesis level (S0-S3), embedding model name, role->model map.
- Global config: providers/credentials location, defaults for new KBs.

Sliders
-------
- SynthesisLevel: S0 index-only, S1 summaries, S2 wiki, S3 atlas.
- RetrievalDepth: R0 lookup, R1 ranked (default), R2 multi-hop, R3 agentic.
Both are small enums used across the package; retrieval code must branch on
what exists in the store, never on the configured S level (docs/architecture.md).

Public API
----------
- class SynthesisLevel(StrEnum): S0..S3, .parse(value); ordered comparison
- class RetrievalDepth(StrEnum): R0..R3, .parse(value); ordered comparison
- DEFAULT_KB_CONFIG: the kb_config rows a new KB starts with
- kb_root() -> Path            (WIKIRAG_HOME, else ./knowledge_bases)
- list_kbs() -> list[str]
- create_kb(name, *, root=None, config=None) -> KnowledgeBase
- open_kb(name, *, root=None) -> KnowledgeBase
- class KnowledgeBase: .name, .root, .store, .config, .db_path, .vectors_dir,
  .raw_dir, .archive_dir, .vault_dir, .synthesis_level, .default_depth,
  .set_config(key, value), .close(); usable as a context manager
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from wikirag.storage.store import Store

WIKIRAG_HOME_ENV = "WIKIRAG_HOME"
DEFAULT_ROOT_NAME = "knowledge_bases"

# Directories of a KB (docs/storage.md "Layout on disk"); kb.sqlite sits beside them.
KB_SUBDIRS = ("vectors", "raw", "archive", "vault")
DB_FILENAME = "kb.sqlite"

_VALID_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class _SliderEnum(StrEnum):
    """Slider levels. Values are ordered lexicographically (S0 < S1 < ...),
    so plain string comparison is the level comparison."""

    @classmethod
    def parse(cls, value: Any) -> Any:
        """Accept a member, or its name in any case ('s2', 'S2')."""
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().upper())
        except ValueError:
            raise ValueError(
                f"unknown {cls.__name__}: {value!r} (expected one of "
                f"{', '.join(m.value for m in cls)})"
            ) from None


class SynthesisLevel(_SliderEnum):
    """Consolidation eagerness, per KB (docs/synthesis.md)."""

    S0 = "S0"  # index only: chunks + FTS + embeddings
    S1 = "S1"  # + document/section summaries
    S2 = "S2"  # + lazy keyword notes and typed links
    S3 = "S3"  # + atlas: overview, hub pages, proactive consolidation


class RetrievalDepth(_SliderEnum):
    """Query depth, per query (docs/retrieval.md)."""

    R0 = "R0"  # lookup: hybrid search + rerank
    R1 = "R1"  # + orientation and the no-answer gate (default)
    R2 = "R2"  # + decomposition, repair loop, consolidation triggers
    R3 = "R3"  # agentic navigation


DEFAULT_KB_CONFIG: dict[str, str] = {
    "synthesis_level": SynthesisLevel.S0.value,
    "default_depth": RetrievalDepth.R1.value,
    # Lazy consolidation defers to background so the triggering query is not
    # blocked on synthesis (D11).
    "consolidation_mode": "defer",
}


def kb_root() -> Path:
    """Root holding every KB directory: $WIKIRAG_HOME, else ./knowledge_bases."""
    env = os.environ.get(WIKIRAG_HOME_ENV)
    return Path(env).expanduser() if env else Path.cwd() / DEFAULT_ROOT_NAME


def _validate_name(name: str) -> str:
    if not _VALID_NAME.match(name):
        raise ValueError(
            f"invalid KB name {name!r}: use letters, digits, '.', '_' or '-', "
            "starting with a letter or digit"
        )
    return name


def list_kbs(root: Path | None = None) -> list[str]:
    """Names of the KBs under the root, sorted. A KB is a dir with kb.sqlite."""
    base = root or kb_root()
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir() if (p / DB_FILENAME).is_file())


def create_kb(
    name: str,
    *,
    root: Path | None = None,
    config: dict[str, str] | None = None,
) -> KnowledgeBase:
    """Create the on-disk layout for a new KB and return its open handle."""
    _validate_name(name)
    base = root or kb_root()
    kb_dir = base / name
    if (kb_dir / DB_FILENAME).exists():
        raise FileExistsError(f"knowledge base {name!r} already exists at {kb_dir}")
    for sub in KB_SUBDIRS:
        (kb_dir / sub).mkdir(parents=True, exist_ok=True)
    store = Store(kb_dir / DB_FILENAME)
    store.migrate()
    with store.transaction():
        for key, value in {**DEFAULT_KB_CONFIG, **(config or {})}.items():
            store.config_set(key, value)
    return KnowledgeBase(name=name, root=kb_dir, store=store)


def open_kb(name: str, *, root: Path | None = None) -> KnowledgeBase:
    """Open an existing KB; migrates its schema if the build is newer."""
    _validate_name(name)
    kb_dir = (root or kb_root()) / name
    if not (kb_dir / DB_FILENAME).is_file():
        raise FileNotFoundError(f"no knowledge base {name!r} at {kb_dir}")
    store = Store(kb_dir / DB_FILENAME)
    store.migrate()
    return KnowledgeBase(name=name, root=kb_dir, store=store)


@dataclass
class KnowledgeBase:
    """Handle passed to every operation: paths + open store + config."""

    name: str
    root: Path
    store: Store

    @property
    def db_path(self) -> Path:
        return self.root / DB_FILENAME

    @property
    def vectors_dir(self) -> Path:
        return self.root / "vectors"

    @property
    def raw_dir(self) -> Path:
        """Inbox: documents waiting for ingest."""
        return self.root / "raw"

    @property
    def archive_dir(self) -> Path:
        """Ingested source documents."""
        return self.root / "archive"

    @property
    def vault_dir(self) -> Path:
        """Exported markdown view; regenerable, safe to delete."""
        return self.root / "vault"

    @property
    def config(self) -> dict[str, str]:
        """Current kb_config rows (a snapshot; write with set_config)."""
        return self.store.config_all()

    def set_config(self, key: str, value: str) -> None:
        self.store.config_set(key, str(value))

    @property
    def synthesis_level(self) -> SynthesisLevel:
        return SynthesisLevel.parse(
            self.store.config_get("synthesis_level", DEFAULT_KB_CONFIG["synthesis_level"])
        )

    @synthesis_level.setter
    def synthesis_level(self, level: SynthesisLevel | str) -> None:
        self.store.config_set("synthesis_level", SynthesisLevel.parse(level).value)

    @property
    def default_depth(self) -> RetrievalDepth:
        return RetrievalDepth.parse(
            self.store.config_get("default_depth", DEFAULT_KB_CONFIG["default_depth"])
        )

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> KnowledgeBase:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
