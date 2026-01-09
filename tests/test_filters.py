import numpy as np
from scipy import signal
import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from filters import design_bandpass, design_lowpass, fs

# ---------- Bandpass tests ----------

def test_bandpass_sos_shape():
    sos = design_bandpass()
    assert sos.ndim == 2
    assert sos.shape[1] == 6


def test_bandpass_passband_gain():
    sos = design_bandpass()
    w, h = signal.sosfreqz(sos, fs=fs)

    passband = (w >= 2425) & (w <= 2575)
    gain_db = 20 * np.log10(np.maximum(np.abs(h[passband]), 1e-12))

    assert np.max(gain_db) > -2   # ≈ 0 dB in passband


def test_bandpass_stopband_attenuation():
    sos = design_bandpass()
    w, h = signal.sosfreqz(sos, fs=fs)

    stopband = (w <= 2300) | (w >= 2700)
    atten_db = 20 * np.log10(np.maximum(np.abs(h[stopband]), 1e-12))

    assert np.max(atten_db) < -35


# ---------- Lowpass tests ----------

def test_lowpass_sos_shape():
    sos = design_lowpass()
    assert sos.ndim == 2
    assert sos.shape[1] == 6


def test_lowpass_passband_gain():
    sos = design_lowpass()
    w, h = signal.sosfreqz(sos, fs=fs)

    passband = w <= 100
    gain_db = 20 * np.log10(np.maximum(np.abs(h[passband]), 1e-12))

    assert np.max(gain_db) > -2


def test_lowpass_stopband_attenuation():
    sos = design_lowpass()
    w, h = signal.sosfreqz(sos, fs=fs)

    stopband = w >= 500
    atten_db = 20 * np.log10(np.maximum(np.abs(h[stopband]), 1e-12))

    assert np.max(atten_db) < -35

def test_fs_positive():
    assert fs > 0

def test_bandpass_stability():
    sos = design_bandpass()
    z, p, k = signal.sos2zpk(sos)
    assert np.all(np.abs(p) < 1)

