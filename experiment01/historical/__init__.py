"""Historical post-P0 analysis modules retained for reproducibility."""

import sys

from . import analysis_artifacts as analysis_artifacts
from . import consolidation_geometry as consolidation_geometry

# Two frozen modules retain their original absolute sibling imports so that
# their byte-level hashes remain unchanged. Register only those sibling names.
sys.modules.setdefault("analysis_artifacts", analysis_artifacts)
sys.modules.setdefault("consolidation_geometry", consolidation_geometry)
