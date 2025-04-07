import numpy as np

def add_noise(sig, sig_snr):
    if sig_snr == 0:
        return sig
    
    noise = np.random.normal(0, 1, sig.shape) * 10**(-sig_snr/20)
    return sig + noise