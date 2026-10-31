import os

# NumPy's maths library starts one thread (and memory buffer) per CPU core. On a
# laptop that is low on memory this can crash Python silently before any test runs.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import sys
from pathlib import Path

# Make `import src...` work when pytest is run from any folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
