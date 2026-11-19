"""A minimal file-based model registry: ``models/<version>/`` plus a ``current`` symlink.

Promoting is an atomic symlink swap, so the service never sees a half-written model and a
rollback is one command. (MLflow's registry does the same job with a database behind it; for
one model and one team a directory is enough and has no moving parts.)
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from taximon import profile as profiles
from taximon.model import ETAModel


def register(
    model: ETAModel,
    train_df: pd.DataFrame,
    zones: dict[int, str],
    root: str | Path = "models",
    version: str | None = None,
) -> Path:
    root = Path(root)
    version = version or datetime.now().strftime("v%Y%m%d-%H%M%S")
    model.meta["version"] = version
    path = model.save(root / version)
    profiles.save(profiles.build(train_df, model.predict(train_df)), path / "profile.json")
    (path / "zones.json").write_text(json.dumps({str(k): v for k, v in zones.items()}))
    return path


def promote(version: str, root: str | Path = "models") -> Path:
    root = Path(root)
    if not (root / version / "model.txt").exists():
        raise FileNotFoundError(f"no model {version} in {root}")
    tmp = root / ".current.tmp"
    tmp.unlink(missing_ok=True)
    tmp.symlink_to(version, target_is_directory=True)
    os.replace(tmp, root / "current")  # atomic on POSIX
    return root / "current"


def versions(root: str | Path = "models") -> list[dict]:
    root = Path(root)
    cur = (root / "current").resolve().name if (root / "current").exists() else None
    out = []
    for p in sorted(root.glob("v*")):
        if (p / "meta.json").exists():
            meta = json.loads((p / "meta.json").read_text())["meta"]
            out.append({"version": p.name, "current": p.name == cur, **meta})
    return out
