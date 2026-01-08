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

    wp_bp = [f/(fs/2) for f in fp_bp]
    ws_bp = [f/(fs/2) for f in fs_bp]

    N_bp, Wn_bp = signal.cheb1ord(wp_bp, ws_bp, gpass_bp, gstop_bp)
    b_bp, a_bp = signal.cheby1(N_bp, gpass_bp, Wn_bp, btype='bandpass')

    y_ch = signal.lfilter(b_bp, a_bp, yr)

    # --- IQ demodulation ---
    yI = 2 * y_ch * np.cos(2 * np.pi * fc * t)
    yQ = 2 * y_ch * np.sin(2 * np.pi * fc * t)

    # --- Lowpass filters (baseband extraction) ---
    fp_lp = 100     # passband edge (Hz)
    fs_lp = 500     # stopband edge (Hz)
    gpass_lp = 1
    gstop_lp = 40

    wp_lp = fp_lp/(fs/2)
    ws_lp = fs_lp/(fs/2)

    N_lp, Wn_lp = signal.buttord(wp_lp, ws_lp, gpass_lp, gstop_lp)
    b_lp, a_lp = signal.butter(N_lp, Wn_lp, btype='lowpass')

    yI_bb = signal.lfilter(b_lp, a_lp, yI)
    yQ_bb = signal.lfilter(b_lp, a_lp, yQ)

    # --- Phase alignment (fix unknown carrier phase) ---
    y_complex = yI_bb + 1j * yQ_bb
    
    mag = np.abs(y_complex)

    gamma = 0.1 * np.max(mag)          # less aggressive than 0.2
    idx = np.where(mag > gamma)[0]

    if len(idx) == 0:
        print("[Rx] No signal detected.")
    else:
        margin = int(0.2 * fs)         # 0.2 s safety margin
        start = max(0, idx[0] - margin)
        end   = min(len(y_complex), idx[-1] + margin)

        y_complex = y_complex[start:end]
        print(f"[Rx] Trimmed segment: {start/fs:.2f}s–{end/fs:.2f}s (len {len(y_complex)/fs:.2f}s)")



    # Estimate average phase (robust enough for this project)
    phi = np.angle(np.mean(y_complex))
    y_aligned = y_complex * np.exp(-1j * phi)

    # Real-valued baseband waveform for decoder
    yb = np.real(y_aligned)


    print(f"[Rx] fs={fs} Hz, BPF order={N_bp}, LPF order={N_lp}")

    # Symbol decoding
    # TODO: Adjust fs (lab 2 only, leave untouched for lab 1 unless you know what you are doing)

    fs = int(1/dt)
    Ns = int(round(Tb * fs))  # samples per bit (should be 200)

    # Try a handful of timing offsets and pick the one that yields most printable text
    def printable_score(s):
        return sum(32 <= ord(c) <= 126 for c in s)

    best = ("", None, -1)  # (string, bits, score)

    # Use the baseband you computed (yb). If it's huge, you can keep it as-is.
    for offset in range(0, Ns, 5):   # step 5 keeps it fast
        # how many full bits fit?
        nbits = (len(yb) - offset) // Ns
        if nbits <= 0:
            continue

        # sample in the middle of each bit
        sample_idx = offset + (np.arange(nbits) * Ns) + Ns//2
        samples = yb[sample_idx]

        # BPSK decision: sign -> bits
        br = (samples > 0).astype(int)

        try:
            s = wcs.decode_string(br)
            sc = printable_score(s)
            if sc > best[2]:
                best = (s, br, sc)
        except Exception:
            pass

        data_rx, br, sc = best
        if br is None:
            br = np.array([], dtype=int)
            data_rx = ""

    print(f"Received: {data_rx} (no of bits: {len(br)}; printable score: {sc}).")



if __name__ == "__main__":    
    main()
