import librosa
from scipy.signal import butter, filtfilt
import numpy as np
import matplotlib.pyplot as plt
import os

file = './Sample4.wav'

def resample_audio(file, target_sr=16000):
    signal, sr = librosa.load(file, sr=target_sr)
    return signal, sr

resample_audio, sr = resample_audio(file)


def butter_lowpass_filter(data, cutoff_freq, sample_rate, order=4):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    print(f"Filtered audio shape: {filtered_data.shape}")
    return filtered_data

filtered_audio = butter_lowpass_filter(resample_audio, cutoff_freq=4000, sample_rate=sr)


def convert_to_model_input(y, target_length):
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]
    return y

model_input = convert_to_model_input(filtered_audio, target_length=16000)
print(f"Model input shape: {model_input.shape}")