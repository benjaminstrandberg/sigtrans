import numpy as np
from scipy import signal
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from parameters import dt, fc



def test_iq_demodulation_shapes():
    fs = int(1/dt)
    t = np.arange(0, 0.1, dt)

    # Fake received sinusoid at carrier
    yr = np.sin(2 * np.pi * fc * t)

    yI = 2 * yr * np.cos(2 * np.pi * fc * t)
    yQ = 2 * yr * np.sin(2 * np.pi * fc * t)

    assert yI.shape == yr.shape
    assert yQ.shape == yr.shape


def test_complex_baseband_generation():
    yI = np.ones(100)
    yQ = np.zeros(100)

    y_complex = yI + 1j * yQ

    assert np.iscomplexobj(y_complex)
    assert np.allclose(np.real(y_complex), 1)
