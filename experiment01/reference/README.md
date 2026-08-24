# Reference analysis stack

This package contains the frozen extraction, held-out screening, accessibility
ladder and consolidation utilities used by Experiment 01 reproduction and
equivalence gates.

`ladder_accessibility.py` is intentionally byte-identical to the frozen source
verified by Phase III (`a34c8574…`). Its original sibling imports are supported
by this package's `__init__.py`; do not reformat that file.

Current Experiment 01 implementation lives one directory above. Reference
utilities can be invoked with `python -m experiment01.reference.<module>`.
