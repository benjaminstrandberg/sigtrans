# fs = 20000 Hz
#
# Bandpass:
#   Passband: 2425–2575 Hz
#   Stopband: <= 2300 Hz, >= 2700 Hz
#   Passband ripple: <= 1 dB
#   Stopband attenuation: >= 40 dB
#
# Lowpass:
#   Passband: 0–100 Hz
#   Stopband: >= 500 Hz
#   Passband ripple: <= 1 dB
#   Stopband attenuation: >= 40 dB

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

fs = 20000

#jag la till de två funtioner för att testa i test_filters.py
def design_bandpass():
    return signal.iirdesign(
        wp=[2425, 2575],
        ws=[2300, 2700],
        gpass=1,
        gstop=40,
        ftype="butter",
        output="sos",
        fs=fs
    )


def design_lowpass():
    return signal.iirdesign(
        wp=100,
        ws=500,
        gpass=1,
        gstop=40,
        ftype="butter",
        output="sos",
        fs=fs
    )

#Bandpass specs
fp_bp = [2425, 2575]   # passband (Hz)
fs_bp = [2300, 2700]   # stopband (Hz)
gpass_bp = 1           # dB
gstop_bp = 40          # dB

sos_bp = signal.iirdesign(
    wp=fp_bp,
    ws=fs_bp,
    gpass=gpass_bp,
    gstop=gstop_bp,
    ftype="butter",
    output="sos",
    fs=fs
)

print("Bandpass SOS sections:", sos_bp.shape[0])

w, h = signal.sosfreqz(sos_bp, worN=4096, fs=fs)

plt.figure()
plt.plot(w, 20*np.log10(np.maximum(np.abs(h), 1e-12)))
plt.axvline(2425, color='r', linestyle='--')
plt.axvline(2575, color='r', linestyle='--')
plt.axvline(2300, color='k', linestyle=':')
plt.axvline(2700, color='k', linestyle=':')
plt.title(f"Bandpass magnitude response (SOS sections {sos_bp.shape[0]})")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude [dB]")
plt.grid(True)
plt.show()

#Lowpass specs 
fp_lp = 100     # passband edge 
fs_lp = 500     # stopband edge 
gpass_lp = 1
gstop_lp = 40

sos_lp = signal.iirdesign(
    wp=fp_lp,
    ws=fs_lp,
    gpass=gpass_lp,
    gstop=gstop_lp,
    ftype="butter",
    output="sos",
    fs=fs
)

print("Lowpass SOS sections:", sos_lp.shape[0])

w, h = signal.sosfreqz(sos_lp, worN=4096, fs=fs)

plt.figure()
plt.plot(w, 20*np.log10(np.maximum(np.abs(h), 1e-12)))
plt.axvline(100, color='r', linestyle='--')
plt.axvline(500, color='k', linestyle=':')
plt.title(f"Lowpass magnitude response (SOS sections {sos_lp.shape[0]})")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude [dB]")
plt.grid(True)
plt.show()
