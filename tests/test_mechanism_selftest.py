"""The Plan C self-test must see every WM_MECHANISMS=all path fire at least once."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import mechanism_selftest


def test_every_mechanism_path_fires():
    counts = mechanism_selftest.run()
    assert set(counts) == set(mechanism_selftest.PATHS)
    assert all(counts.values()), {p: c for p, c in counts.items() if not c}
