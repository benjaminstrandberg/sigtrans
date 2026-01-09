#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Receiver template for the wireless communication system project in Signals and
transforms

2022-present -- Roland Hostettler <roland.hostettler@angstrom.uu.se>
"""

import argparse
import numpy as np
from scipy import signal
import sounddevice as sd

import wcslib as wcs

# TODO: Add relevant parameters to parameters.py
from parameters import Tb, dt, fc

def main():
    parser = argparse.ArgumentParser(
        prog='receiver',
        description='Acoustic wireless communication system -- receiver.'
    )
    parser.add_argument(
        '-d',
        '--duration',
        help='receiver recording duration',
        type=float,
        default=10
    )
    args = parser.parse_args()

    # Set parameters
    T = args.duration

    # Receive signal
    print(f'Receiving for {T} s.')
    yr = sd.rec(int(T/dt), samplerate=1/dt, channels=1, blocking=True)
    yr = yr[:, 0]           # Remove second channel

    # TODO: Implement demodulation, etc. here
    fs = int(1/dt)
    t = np.arange(len(yr)) * dt

    # --- Rx bandpass filter (interference/noise rejection) ---
    fp_bp = [2425, 2575]   # passband (Hz)
    fs_bp = [2300, 2700]   # stopband (Hz)
    gpass_bp = 1           # dB
    gstop_bp = 40          # dB

    # Design bandpass as Butterworth from requirements, using SOS for stability
    sos_bp = signal.iirdesign(
        wp=fp_bp,          # Hz (because fs=fs is provided)
        ws=fs_bp,          # Hz
        gpass=gpass_bp,    # dB
        gstop=gstop_bp,    # dB
        ftype="butter",
        output="sos",
        fs=fs
    )

    y_ch = signal.sosfilt(sos_bp, yr)


    # --- IQ demodulation ---
    yI = 2 * y_ch * np.cos(2 * np.pi * fc * t)
    yQ = 2 * y_ch * np.sin(2 * np.pi * fc * t)

    # --- Lowpass filters (baseband extraction) ---
    fp_lp = 100     # passband edge (Hz)
    fs_lp = 500     # stopband edge (Hz)
    gpass_lp = 1
    gstop_lp = 40

    # Design lowpass as Butterworth from requirements, using SOS for stability
    sos_lp = signal.iirdesign(
        wp=fp_lp,          # Hz
        ws=fs_lp,          # Hz
        gpass=gpass_lp,    # dB
        gstop=gstop_lp,    # dB
        ftype="butter",
        output="sos",
        fs=fs
    )

    yI_bb = signal.sosfilt(sos_lp, yI)
    yQ_bb = signal.sosfilt(sos_lp, yQ)


    # --- Phase alignment (fix unknown carrier phase) ---
    y_complex = yI_bb + 1j * yQ_bb
    
    win_sec = 6.0
    win = int(win_sec * fs)
    y_complex = y_complex[:win]
    
    mag = np.abs(y_complex)
    gamma = 0.1 * np.max(mag)          # threshold (20% of max)
    start = np.argmax(mag > gamma)     # first index above threshold
    y_complex = y_complex[start:]      # trim everything before start


    # Estimate average phase (robust enough for this project)
    phi = np.angle(np.mean(y_complex))
    y_aligned = y_complex * np.exp(-1j * phi)

    # Real-valued baseband waveform for decoder
    yb = np.real(y_aligned)


    print(f"[Rx] fs={fs} Hz, BPF sections={sos_bp.shape[0]}, LPF sections={sos_lp.shape[0]}")

    # Symbol decoding
    # TODO: Adjust fs (lab 2 only, leave untouched for lab 1 unless you know what you are doing)
    br = wcs.decode_baseband_signal(yb, Tb, 1/dt)
    data_rx = wcs.decode_string(br)
    print(f'Received: {data_rx} (no of bits: {len(br)}).')


if __name__ == "__main__":    
    main()
