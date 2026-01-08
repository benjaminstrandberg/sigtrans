# FINAL FILTER DESIGN
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


#BANDPASS

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

fs = 20000

# ----- Bandpass specs -----
fp_bp = [2425, 2575]   # passband (Hz)
fs_bp = [2300, 2700]   # stopband (Hz)
gpass_bp = 1           # dB
gstop_bp = 40          # dB

wp_bp = [f/(fs/2) for f in fp_bp]
ws_bp = [f/(fs/2) for f in fs_bp]

N_bp, Wn_bp = signal.cheb1ord(wp_bp, ws_bp, gpass_bp, gstop_bp)
b_bp, a_bp = signal.cheby1(N_bp, gpass_bp, Wn_bp, btype='bandpass')

print("Bandpass order:", N_bp)


w, h = signal.freqz(b_bp, a_bp, fs=fs)

plt.figure()
plt.plot(w, 20*np.log10(np.maximum(np.abs(h), 1e-12)))
plt.axvline(2400, color='r', linestyle='--')
plt.axvline(2600, color='r', linestyle='--')
plt.axvline(2300, color='k', linestyle=':')
plt.axvline(2700, color='k', linestyle=':')
plt.title(f"Bandpass magnitude response (order {N_bp})")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude [dB]")
plt.grid(True)
plt.show()



# ----- Lowpass specs -----
fp_lp = 100     # passband edge (Hz)
fs_lp = 500     # stopband edge (Hz)
gpass_lp = 1
gstop_lp = 40

wp_lp = fp_lp/(fs/2)
ws_lp = fs_lp/(fs/2)

N_lp, Wn_lp = signal.buttord(wp_lp, ws_lp, gpass_lp, gstop_lp)
b_lp, a_lp = signal.butter(N_lp, Wn_lp, btype='lowpass')

print("Lowpass order:", N_lp)


w, h = signal.freqz(b_lp, a_lp, fs=fs)

plt.figure()
plt.plot(w, 20*np.log10(np.maximum(np.abs(h), 1e-12)))
plt.axvline(100, color='r', linestyle='--')
plt.axvline(500, color='k', linestyle=':')
plt.title(f"Lowpass magnitude response (order {N_lp})")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude [dB]")
plt.grid(True)
plt.show()

