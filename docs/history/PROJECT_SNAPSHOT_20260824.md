# Project snapshot — 2026-08-24

This record accompanies the Git snapshot tag
`project-snapshot-2026-08-24-experiment01`.

## Scope

The snapshot records the complete Git-visible working-tree state: tracked
changes and deletions, plus every non-ignored source, test, protocol and report
file present at snapshot time.

Bulk runtime assets remain governed by `.gitignore` and are not duplicated in
Git. The workspace occupies approximately 391 GB because it contains datasets,
checkpoints and validation outputs; embedding those objects in the repository
would not create a practical source snapshot.

Pre-snapshot parent commit:
`6a94bd5890539037a00fa7f635776707ed647183`.

Working-tree inventory before staging:

- modified tracked files: 1;
- deleted tracked files: 417;
- new non-ignored files: 152;
- total size of new non-ignored files: approximately 4.9 MB.

## Frozen scientific artifact reference

Experiment 01 predictability-allocation run:

`validation/experiment01/predictability_allocation_20260819/run`

- execution status: complete;
- runtime: 17.453550815582275 seconds;
- failure rows: 0;
- preregistered outcome: `fail`;
- run manifest file SHA-256:
  `31d348cee4374a8ee7cdd29d6d578b60a99b5f0dabca2a374a991adecfc84e61`;
- run manifest canonical payload SHA-256:
  `6b3c1bafe9e5687631d9dc9fad7a1534a957e3fc197bf8126be9d49bc06d40b9`;
- frozen protocol file SHA-256:
  `18e67f04dfa1e3418a333966ac2ce0629feb0670c50a48edf3800e38cb338ad1`;
- frozen input inventory SHA-256:
  `7ddd24783f483f4f8c827881b892083e753fafe010bb64c2fa4fd2607b645357`;
- source dataset SHA-256:
  `7617dbbfcee56377f980a606267397861f6613017f0a2aca1e218407726ef862`.

The run manifest contains the SHA-256 and size of every scientific output and
cache artifact. It was revalidated after completion with zero mismatches.

## Verification state

The Experiment 01 test suite, including Phase I, Phase II, Phase III and the
predictability-allocation diagnostic, last completed with 102 passing tests.
