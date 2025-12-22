import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

fs = 40000
order = 5

def bandpass_filter(x,fs):    
    b,a = signal.butter(5, [2400 / (fs / 2),2600 / (fs/2)], btype='bandpass')
    
    return signal.lfilter(b,a,x)

def lowpass_filter(x, fs):
    b, a = signal.butter(5, 200 / (fs / 2),btype='lowpass')
    return signal.lfilter(b, a, x)
 


b_bp, a_bp = signal.butter(
    order,
    [2400 / (fs / 2), 2600 / (fs / 2)],
    btype='bandpass'
)

w_bp, h_bp = signal.freqz(b_bp, a_bp, fs=fs) 

plt.figure()
plt.plot(w_bp, 20 * np.log10(np.abs(h_bp)))
plt.title("Bandpass filter magnitude response")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude [dB]")
plt.grid()
plt.show()    


b_lp, a_lp = signal.butter(
    order,
    200 / (fs / 2),
    btype='lowpass'
)

w_lp, h_lp = signal.freqz(b_lp, a_lp, fs=fs)

plt.figure()
plt.plot(w_lp, 20 * np.log10(np.abs(h_lp)))
plt.title("Lowpass filter magnitude response")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude [dB]")
plt.grid()
plt.show()