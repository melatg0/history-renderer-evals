"""Parse an experiment YAML into an enumerated list of cells + provenance.

A *cell* is one (history_level, script_id) combination run as a single
`inspect eval` invocation with its own ``--log-dir`` and epoch count. This
module is pure Python (no inspect_ai dependency) so cell enumeration and the
cost estimate can be unit-tested without the eval harness.

See configs/t1_1_history_robustness.yaml for the schema.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    """Repository root (this file is eval/agentic_misalignment/experiment/)."""
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Cell:
    """One runnable cell of the experiment."""

    level: str
    script_id: str | None  # None for non-scripted levels (A_raw, A0)
    n: int

    @property
    def cell_key(self) -> str:
        """Stable identifier, e.g. 'A3/script_04' or 'A_raw'."""
        return f"{self.level}/{self.script_id}" if self.script_id else self.level

    def log_dir(self, log_base: Path) -> Path:
        return log_base / self.level / self.script_id if self.script_id else log_base / self.level


@dataclass
class ExperimentConfig:
    path: Path
    raw: dict[str, Any]

    @property
    def experiment_id(self) -> str:
        return self.raw["experiment_id"]

    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", 0))

    @property
    def model(self) -> str:
        return self.raw["model"]

    @property
    def grader_model(self) -> str | None:
        return self.raw.get("grader_model")

    @property
    def temperature(self) -> float:
        return float(self.raw.get("temperature", 1.0))

    @property
    def max_connections(self) -> int:
        return int(self.raw.get("max_connections", 10))

    @property
    def reasoning_effort(self) -> str | None:
        """Current-session reasoning control, held constant across cells.

        Passed to `inspect eval --reasoning-effort` so the only reasoning that
        varies between conditions is the injected prior-session trace.
        None => omit the flag (provider default). Valid: none|minimal|low|medium|high|xhigh.
        """
        v = self.raw.get("reasoning_effort")
        return str(v) if v is not None else None

    @property
    def common(self) -> dict[str, Any]:
        return dict(self.raw.get("common", {}))

    @property
    def cost(self) -> dict[str, Any]:
        return dict(self.raw.get("cost", {}))

    @property
    def log_base(self) -> Path:
        return self._resolve(self.raw.get("log_base", f"logs/{self.experiment_id}"))

    @property
    def results_root(self) -> Path:
        return self._resolve(self.raw.get("results_root", "results"))

    def _resolve(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else repo_root() / path

    def cells(self) -> list[Cell]:
        """Enumerate every (level, script_id) cell with its rollout count."""
        cells: list[Cell] = []
        for level, spec in self.raw["levels"].items():
            spec = spec or {}
            scripts = spec.get("scripts") or []
            if scripts:
                n_per = int(spec["n_per_script"])
                for sid in scripts:
                    cells.append(Cell(level=level, script_id=sid, n=n_per))
            else:
                cells.append(Cell(level=level, script_id=None, n=int(spec["n"])))
        return cells

    def total_rollouts(self, dry_run: bool = False) -> int:
        if dry_run:
            return len(self.cells())
        return sum(c.n for c in self.cells())

    def estimate_usd(self, dry_run: bool = False) -> float:
        avg_tokens = float(self.cost.get("avg_tokens_per_rollout", 0))
        price = float(self.cost.get("usd_per_1k_tokens", 0))
        return self.total_rollouts(dry_run) * avg_tokens / 1000.0 * price

    @property
    def max_usd(self) -> float | None:
        v = self.cost.get("max_usd")
        return float(v) if v is not None else None

    def config_hash(self) -> str:
        """SHA-256 of the raw config file bytes (provenance)."""
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        p = Path(path)
        if not p.is_absolute():
            p = (repo_root() / p) if not p.exists() else p.resolve()
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        return cls(path=p.resolve(), raw=raw)


def sanitize_model(model: str) -> str:
    """Filesystem-safe model string, e.g. anthropic__claude-opus-4-1-...."""
    return model.replace("/", "__").replace(":", "_")
