# Experiment 01 canonical checkpoints

## Distribution decision

Distribute the **nine canonical epoch-20 checkpoints**, not the complete local
training directory:

| content | files | size | distribution |
|---|---:|---:|---|
| canonical `3 arms × 3 seeds` | 9 | 84,199,395 bytes | release artifact |
| all epochs plus aliases/history | 210 | about 1.8 GiB | local only |

The portable file-level contract is
[`CHECKPOINTS_MULTISEED_MANIFEST.json`](CHECKPOINTS_MULTISEED_MANIFEST.json).
Every record declares arm, seed, epoch, relative path, byte size and SHA-256.

## Deterministic release archive

From the repository root:

```bash
mkdir -p dist
python -m scripts.artifacts.package_experiment01_checkpoints \
  --out dist/experiment01_canonical_checkpoints_ep020.tar
```

Expected output:

```text
archive name    experiment01_canonical_checkpoints_ep020.tar
archive bytes   84,213,760
archive SHA-256 3e268b6fa73a122399e4b420e989a4d37112e2696efe55b4bf095892ab82ed06
manifest SHA-256 5d416d0d87cbb20f097afaaa85383d8838fb9abb9233aade29afc0ed3a2fad11
```

The packager fails closed on a missing file, unexpected size, SHA mismatch,
duplicate path, wrong file count or wrong total checkpoint bytes. The archive
normalizes ownership, permissions and timestamps, making its hash reproducible.

The archive should be uploaded as a release/research artifact and linked here;
it should not be committed as an ordinary Git blob. A public download URL has
not yet been assigned.
