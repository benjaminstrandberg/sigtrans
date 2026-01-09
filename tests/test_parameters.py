
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import parameters as p


def test_sampling_parameters():
    assert p.fs > 0
    assert p.dt > 0
    assert abs(p.dt - 1/p.fs) < 1e-12

def test_signal_parameters():
    assert p.fc > 0
    assert p.Tb > 0
    assert p.Ac > 0
