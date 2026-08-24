# Historical checkpoint compatibility

These files preserve the class definitions and historical split behavior used
by exploratory checkpoints created before the controlled multiseed protocol.
Several scripts under `scripts/evaluation/` import them to load those artifacts.

They are not current training entrypoints. For new runs use the modules one
directory above, which fix split-seed parity and endpoint parity across arms.
