# Historical post-P0 analysis stack

This package contains the corrected post-P0 extraction, held-out screening,
accessibility ladder and consolidation utilities that predate Experiment 01.
They remain active only as historical reproduction and equivalence gates.

`ladder_accessibility.py` is intentionally byte-identical to the frozen source
verified by Phase III (`a34c8574…`). Its original sibling imports are supported
by this package's `__init__.py`; do not reformat that file.

Current Experiment 01 implementation lives one directory above. Historical
commands should be invoked with `python -m experiment01.historical.<module>`.
