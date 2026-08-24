from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.artifacts.package_experiment01_checkpoints import package


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(root: Path) -> Path:
    records = []
    total = 0
    for index in range(9):
        relative = Path("checkpoints") / f"seed{index}" / "epoch_020.pt"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = f"checkpoint-{index}".encode()
        path.write_bytes(payload)
        total += len(payload)
        records.append(
            {
                "path": str(relative),
                "size_bytes": len(payload),
                "sha256": _sha256(path),
            }
        )
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "canonical_size_bytes": total,
                "checkpoints": records,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return manifest


def test_checkpoint_archive_is_deterministic(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    first = package(tmp_path, manifest, tmp_path / "first.tar")
    second = package(tmp_path, manifest, tmp_path / "second.tar")
    assert first["file_count"] == 9
    assert first["archive_sha256"] == second["archive_sha256"]


def test_checkpoint_packager_fails_closed_on_tampering(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    checkpoint = tmp_path / "checkpoints/seed4/epoch_020.pt"
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="size mismatch|SHA-256 mismatch"):
        package(tmp_path, manifest, tmp_path / "invalid.tar")
