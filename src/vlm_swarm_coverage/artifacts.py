"""Run artifacts: a directory per run holding the config snapshot, provenance, and an event log.

Every metric in this project is computed offline from a logged run, so that a metric can change
without re-running a sweep. That makes the event log the primary output of a run and the run
directory the unit of reproducibility: config + seed + code version in, events out.

Layout of one run directory:

    runs/<name>-<timestamp>/
        config.json    the RunConfig that produced this run, verbatim
        meta.json      provenance: run id, package version, git sha, creation time
        events.jsonl   one JSON object per line, append-only, time-ordered per writer
"""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vlm_swarm_coverage import __version__
from vlm_swarm_coverage.config import RunConfig

if TYPE_CHECKING:
    from collections.abc import Iterator

log = logging.getLogger(__name__)

CONFIG_FILE = "config.json"
META_FILE = "meta.json"
EVENTS_FILE = "events.jsonl"


def git_sha(repo: Path | None = None) -> str | None:
    """The current commit, or None when not in a git checkout. The absence is logged, not hidden."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        log.warning("git sha unavailable, provenance will be incomplete: %s", exc)
        return None
    return out.stdout.strip()


class EventLog:
    """Append-only JSONL writer. Each event carries a kind, a sim time, and a free payload.

    Flushes every `flush_every` events so a crashed run still leaves readable output, and always
    on close. Use as a context manager.
    """

    def __init__(self, path: Path, flush_every: int = 100) -> None:
        self.path = path
        self._fh = path.open("a", encoding="utf-8")
        self._flush_every = flush_every
        self._since_flush = 0
        self.count = 0

    def write(self, kind: str, t: float, **payload: Any) -> None:
        record = {"kind": kind, "t": t, **payload}
        self._fh.write(json.dumps(record, separators=(",", ":")) + "\n")
        self.count += 1
        self._since_flush += 1
        if self._since_flush >= self._flush_every:
            self.flush()

    def flush(self) -> None:
        self._fh.flush()
        self._since_flush = 0

    def close(self) -> None:
        if not self._fh.closed:
            self.flush()
            self._fh.close()

    def __enter__(self) -> EventLog:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class RunDir:
    """A run's directory on disk. `create` starts a new run; `open` reads an existing one."""

    def __init__(self, path: Path, config: RunConfig, meta: dict[str, Any]) -> None:
        self.path = path
        self.config = config
        self.meta = meta

    @classmethod
    def create(cls, root: Path | str, config: RunConfig, repo: Path | None = None) -> RunDir:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
        path = Path(root) / f"{config.name}-{stamp}"
        path.mkdir(parents=True, exist_ok=False)
        meta = {
            "run_id": path.name,
            "created_at": datetime.now(UTC).isoformat(),
            "package_version": __version__,
            "git_sha": git_sha(repo),
        }
        config.dump_json(path / CONFIG_FILE)
        (path / META_FILE).write_text(json.dumps(meta, indent=2) + "\n")
        log.info("run directory created: %s", path)
        return cls(path, config, meta)

    @classmethod
    def open(cls, path: Path | str) -> RunDir:
        path = Path(path)
        config_path = path / CONFIG_FILE
        if not config_path.exists():
            raise FileNotFoundError(
                f"{path} is not a run directory: missing {CONFIG_FILE}. "
                f"Pass the directory that holds config.json, meta.json and events.jsonl."
            )
        config = RunConfig.load(config_path)
        meta = json.loads((path / META_FILE).read_text())
        return cls(path, config, meta)

    @property
    def events_path(self) -> Path:
        return self.path / EVENTS_FILE

    def events(self, flush_every: int = 100) -> EventLog:
        return EventLog(self.events_path, flush_every=flush_every)

    def iter_events(self, kind: str | None = None) -> Iterator[dict[str, Any]]:
        """Stream logged events, optionally filtered by kind. Empty if the run never logged."""
        if not self.events_path.exists():
            return
        with self.events_path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                record = json.loads(line)
                if kind is None or record["kind"] == kind:
                    yield record
