#!/usr/bin/env python3
"""Verify and package the nine canonical Experiment 01 checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile


DEFAULT_MANIFEST = Path(
    "docs/experiment01/CHECKPOINTS_MULTISEED_MANIFEST.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_tar_info(path: Path, archive_name: str) -> tarfile.TarInfo:
    info = tarfile.TarInfo(archive_name)
    info.size = path.stat().st_size
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    return info


def verify_manifest(root: Path, manifest: dict) -> list[tuple[Path, str]]:
    records = manifest.get("checkpoints")
    if not isinstance(records, list) or len(records) != 9:
        raise RuntimeError("manifest must contain exactly nine checkpoints")
    verified: list[tuple[Path, str]] = []
    total = 0
    seen: set[str] = set()
    for record in records:
        relative = str(record["path"])
        if relative in seen:
            raise RuntimeError(f"duplicate checkpoint path: {relative}")
        seen.add(relative)
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        if size != int(record["size_bytes"]):
            raise RuntimeError(f"size mismatch: {relative}")
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch: {relative}")
        total += size
        verified.append((path, relative))
    if total != int(manifest["canonical_size_bytes"]):
        raise RuntimeError("canonical checkpoint byte total differs")
    return verified


def package(root: Path, manifest_path: Path, output: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verified = verify_manifest(root, manifest)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, mode="w", format=tarfile.PAX_FORMAT) as archive:
        info = _normalized_tar_info(
            manifest_path, "CHECKPOINTS_MULTISEED_MANIFEST.json"
        )
        with manifest_path.open("rb") as handle:
            archive.addfile(info, handle)
        for path, relative in verified:
            info = _normalized_tar_info(path, relative)
            with path.open("rb") as handle:
                archive.addfile(info, handle)
    return {
        "status": "complete",
        "path": str(output.resolve()),
        "file_count": len(verified),
        "checkpoint_bytes": sum(path.stat().st_size for path, _ in verified),
        "archive_bytes": output.stat().st_size,
        "archive_sha256": sha256_file(output),
        "manifest_sha256": sha256_file(manifest_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = (
        args.manifest
        if args.manifest.is_absolute()
        else root / args.manifest
    )
    print(
        json.dumps(
            package(root, manifest_path.resolve(), args.out.resolve()),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
