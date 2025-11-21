# eval/structural.py
import numpy as np
from scipy.signal import welch


def acf_1d(x, max_lag):
    """
    x: 1D array
    returns: [max_lag + 1] ACF values
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    result = np.correlate(x, x, mode="full")
    result = result[result.size // 2:]
    result = result / (result[0] + 1e-12)
    return result[: max_lag + 1]


def acf_deviation(x, x_rec, max_lag):
    """
    x, x_rec: 1D arrays
    returns: mse between acf vectors, and the two acf vectors
    """
    acf_orig = acf_1d(x, max_lag)
    acf_rec = acf_1d(x_rec, max_lag)
    dev = np.mean((acf_orig - acf_rec) ** 2)
    return dev, acf_orig, acf_rec


def spectrum_1d(x, fs=1.0):
    """
    x: 1D array
    returns: (freqs, power) from Welch
    """
    x = np.asarray(x, dtype=np.float64)
    f, Pxx = welch(x, fs=fs, nperseg=min(256, len(x)))
    return f, Pxx


def spectrum_deviation(x, x_rec, fs=1.0, eps=1e-8):
    f1, S1 = spectrum_1d(x, fs)
    f2, S2 = spectrum_1d(x_rec, fs)
    log1 = np.log(S1 + eps)
    log2 = np.log(S2 + eps)
    dev = np.mean((log1 - log2) ** 2)
    return dev, (f1, log1), (f2, log2)

