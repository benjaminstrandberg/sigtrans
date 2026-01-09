#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transmitter template for the wireless communication system project in Signals and
transforms

For plain text inputs, run:
$ python3 transmitter.py "Hello World!"

For binary inputs, run:
$ python3 transmitter.py -b 010010000110100100100001

2022-present -- Roland Hostettler <roland.hostettler@angstrom.uu.se>
"""

import argparse
import numpy as np
from scipy import signal
import sounddevice as sd

import wcslib as wcs

# TODO: Add relevant parameters to parameters.py
from parameters import Tb, dt, fc, Ac, fs


def main():
    print("hello")
    parser = argparse.ArgumentParser(
        prog='transmitter',
        description='Acoustic wireless communication system -- transmitter.'
    )
    parser.add_argument(
        '-b',
        '--binary',
        help='message is a binary sequence',
        action='store_true'
    )
    parser.add_argument('message', help='message to transmit', nargs='?')
    args = parser.parse_args()

    if args.message is None:
        args.message = 'Hello World!'

    # Set parameters
    data = args.message

    # Convert string to bit sequence or string bit sequence to numeric bit
    # sequence
    if args.binary:
        bs = np.array([bit for bit in map(int, data)])
    else:
        bs = wcs.encode_string(data)
    
    # Transmit signal
    print(f'Sending: {data} (no of bits: {len(bs)}; message duration: {np.round(len(bs)*Tb, 1)} s).')

    # Encode baseband signal
    # TODO: Adjust fs (lab 2 only, leave untouched for lab 1 unless you know what you are doing)
    xb = wcs.encode_baseband_signal(bs, Tb, 1/dt)

    # TODO: Implement transmitter code here
    t = np.arange(len(xb)) * dt
    
    xm = Ac * xb * np.sin(2 * np.pi * fc * t)
    
    # Use actual sampling rate (must match sounddevice playback)
    fs_local = int(1/dt)

    fp_bp = [2425, 2575]   # passband (Hz)
    fs_bp = [2300, 2700]   # stopband (Hz)
    gpass_bp = 1           # dB ripple
    gstop_bp = 40          # dB attenuation

    # Requirements-driven Butterworth BPF, implemented as SOS
    sos_bp = signal.iirdesign(
        wp=fp_bp,
        ws=fs_bp,
        gpass=gpass_bp,
        gstop=gstop_bp,
        ftype="butter",
        output="sos",
        fs=fs_local
    )

    # Filter transmit signal (band-limit into allocated channel)
    xt = signal.sosfilt(sos_bp, xm)

    print(f"[Tx] fs={fs_local} Hz, BPF sections={sos_bp.shape[0]}, duration={len(xt)/fs_local:.2f} s")

    
    print("[Tx] xb unique values:", np.unique(xb)[:10])


    # Ensure the signal is mono, then play through speakers
    xt = np.stack((xt, np.zeros(xt.shape)), axis=1)
    sd.play(xt, 1/dt, blocking=True)


if __name__ == "__main__":    
    main()
