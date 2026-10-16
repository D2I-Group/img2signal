#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.signal import ShortTimeFFT
from .utils.plotter import TimeSeriesPlotter

class STFT_Plotter(TimeSeriesPlotter):
    '''
    STFT Plotter: Plots the Short-Time Fourier Transform of a univariate time series signal.

    Usage:
    plotter = STFT_Plotter()
    plotter.plot(x, window_size=20, hop=10, T_x=3600)

    Args:
    - x: np.ndarray
    - window_size
    - hop: int
    - T_x: float
    - save_file: str
    - label: bool
    '''
    def __init__(self):
        super().__init__()
    
    def plot(self, x: np.ndarray, window_size: int = 10, hop: int = 1, 
             T_x: float = 3600, **kwargs):
        def STFT(
            x: np.ndarray,
            window_size: int,
            hop: int,
            T_x: float,
            *,
            window_type: str = "hann",
            beta: int = 14
        ):
            if window_type == "rectangular":
                w = np.ones(window_size)
            elif window_type == "hann":
                w = np.hanning(window_size)
            elif window_type == "hamming":
                w = np.hamming(window_size)
            elif window_type == "blackman":
                w = np.blackman(window_size)
            elif window_type == "kaiser":
                w = np.kaiser(window_size, beta)
            else:
                raise ValueError(f"Invalid window type: {window_type}")

            N = len(x)
            SFT = ShortTimeFFT(w, hop=hop, fs=1 / T_x)
            Sx = SFT.stft(x)
            return Sx
        
        return STFT(x, window_size=window_size, hop=hop, T_x=T_x, **kwargs)


class Wavelet_Plotter(TimeSeriesPlotter):
    '''
    Wavelet Plotter: Plots the Continuous Wavelet Transform of a univariate time series signal.

    Usage:
    plotter = Wavelet_Plotter()
    plotter.plot(x, scales=np.arange(1, 400), save_file='./wavelet.pdf')

    Args:
    - x: np.ndarray
    - scales: np.ndarray
    - save_file: str
    - label: bool
    '''
    def __init__(self):
        super().__init__()
    
    def plot(self, x: np.ndarray, scales: np.ndarray = np.arange(1, 400), **kwargs):
        def wavelet(
            x: np.ndarray,
            scales: np.ndarray = np.arange(1, 400),
            *,
            wavelet: str = "morl",
        ):
            x = x.detach().numpy()
            coefficients, frequencies = pywt.cwt(x, scales, wavelet)
            return coefficients
        return wavelet(x, scales=scales, **kwargs)
    


class WaveletForecast_Plotter(TimeSeriesPlotter):
    '''
    Wavelet Plotter: Plots the Continuous Wavelet Transform of a univariate time series signal.

    Usage:
    plotter = Wavelet_Plotter()
    plotter.plot(x, scales=np.arange(1, 400), save_file='./wavelet.pdf')

    Args:
    - x: np.ndarray
    - scales: np.ndarray
    - save_file: str
    - label: bool
    '''
    def __init__(self):
        super().__init__()
    
    def plot(self, x: np.ndarray, scales: np.ndarray = np.arange(1, 400), **kwargs):
        def wavelet(
            x: np.ndarray,
            scales: np.ndarray = np.arange(1, 400),
            *,
            wavelet: str = "morl",
        ):
            coefficients, frequencies = pywt.cwt(x, scales, wavelet)
            return coefficients
        return wavelet(x, scales=scales, **kwargs)


class Filterbank_Plotter(TimeSeriesPlotter):
    '''
    Filterbank Plotter: Plots the Filterbank of a univariate time series signal.

    Usage:
    plotter = Filterbank_Plotter()
    plotter.plot(x, window_size=20, hop=10, T_x=3600, num_filters=10, save_file='./filterbank.pdf')

    Args:
    - x: np.ndarray
    - window_size: int
    - hop: int
    - T_x: float
    - num_filters: int
    - save_file: str
    - label: bool
    '''

    def __init__(self):
        super().__init__()
    
    def plot(self, x: np.ndarray, window_size: int = 10, hop: int = 1, 
             T_x: float = 3600, num_filters: int = 10, **kwargs):

        def filterbank(
            x: np.ndarray,
            window_size: int,
            hop: int,
            T_x: float,
            num_filters: int = 10,
            *,
            window_type: str = "hann",
            beta: int = 14,
            use_mel: bool = False
        ):
            if window_type == "rectangular":
                w = np.ones(window_size)
            elif window_type == "hann":
                w = np.hanning(window_size)
            elif window_type == "hamming":
                w = np.hamming(window_size)
            elif window_type == "blackman":
                w = np.blackman(window_size)
            elif window_type == "kaiser":
                w = np.kaiser(window_size, beta)
            else:
                raise ValueError(f"Invalid window type: {window_type}")

            N = len(x)
            SFT = ShortTimeFFT(w, hop=hop, fs=1 / T_x)
            Sx = SFT.stft(x)

            # Calculate frequency range
            freqs = np.fft.rfftfreq(window_size, T_x)

            if use_mel:
                # Create mel-spaced filters
                mel_min = 2595 * np.log10(1 + freqs[0] / 700)
                mel_max = 2595 * np.log10(1 + freqs[-1] / 700)
                mel_points = np.linspace(mel_min, mel_max, num_filters + 2)
                freq_points = 700 * (10 ** (mel_points / 2595) - 1)

                # Create triangular filters
                filter_bank = np.zeros((num_filters, len(freqs)))
                for i in range(num_filters):
                    f_left = freq_points[i]
                    f_center = freq_points[i + 1]
                    f_right = freq_points[i + 2]

                    # Create triangular filter
                    left_slope = (freqs - f_left) / (f_center - f_left)
                    right_slope = (f_right - freqs) / (f_right - f_center)
                    filter_bank[i] = np.maximum(0, np.minimum(left_slope, right_slope))
            else:
                # Create linear-spaced filters
                bandwidth = freqs[-1] / num_filters
                filter_bank = np.zeros((num_filters, len(freqs)))
                for i in range(num_filters):
                    center = (i + 0.5) * bandwidth
                    filter_bank[i] = np.exp(-0.5 * ((freqs - center) / (bandwidth / 2)) ** 2)

            # Apply the filter bank to the STFT result
            filtered_spectrogram = np.dot(filter_bank, np.abs(Sx) ** 2)
            return filtered_spectrogram

        return filterbank(x, window_size=window_size, hop=hop, T_x=T_x, num_filters=num_filters, **kwargs)