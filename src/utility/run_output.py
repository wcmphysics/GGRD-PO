"""Publish completed stage output under a fresh directory."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator


@contextmanager
def stage_output(destination: Path) -> Iterator[Path]:
    """Keep partial runs out of the destination and refuse existing paths."""
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Output path {destination} exists. Expected a fresh directory.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{destination.name}.stage-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "artifacts"
        staging.mkdir()
        yield staging
        staging.rename(destination)


def file_fingerprint(path: Path) -> str:
    """Return the SHA-256 of an input file."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, payload: object) -> None:
    """Write portable JSON with explicit nulls and finite scalars."""
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
