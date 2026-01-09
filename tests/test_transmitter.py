import numpy as np
import pytest
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from parameters import dt, fc, Ac

def test_modulation_output_shape():
    xb = np.ones(100)
    t = np.arange(len(xb)) * dt

    xm = Ac * xb * np.sin(2 * np.pi * fc * t)

    assert xm.shape == xb.shape


def test_modulation_amplitude():
    xb = np.ones(100)
    t = np.arange(len(xb)) * dt

    xm = Ac * xb * np.sin(2 * np.pi * fc * t)

    assert np.max(np.abs(xm)) <= Ac + 1e-6

def test_modulation_handles_zero_bits():
    xb = np.array([0, 1, 0, 1, 1, 0])
    t = np.arange(len(xb)) * dt

    xm = Ac * xb * np.sin(2 * np.pi * fc * t)

    # When xb = 0, xm must be exactly 0
    assert np.allclose(xm[xb == 0], 0, atol=1e-12)
