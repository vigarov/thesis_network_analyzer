from pathlib import Path
import sys

_APP_DIR = Path(__file__).resolve().parent
_PROJECT = _APP_DIR.parent
RESULTS = _PROJECT / "results"

if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

# Eval batch for MNIST-style experiments: 10 digits × 5 samples (see experiments/mnist/base.py).
_NETWORK_N_DIGITS = 10
_NETWORK_SAMPLES_PER_DIGIT = 5
_NETWORK_EXPECTED_BATCH = _NETWORK_N_DIGITS * _NETWORK_SAMPLES_PER_DIGIT
