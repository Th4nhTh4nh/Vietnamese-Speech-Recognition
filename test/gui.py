import tkinter as tk
from tkinter import filedialog
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav

class Recorder:
    def __init__(self, fs=44100):
        self.fs = fs
        self.isrecording = False

    def start_recording(self):
        self.isrecording = True
        print("Recording started")
        self.audio = sd.rec(int(self.fs), samplerate=self.fs, channels=2)

    def stop_recording(self):
        if self.isrecording:
            print("Recording stopped")
            self.isrecording = False
            sd.stop()
            return self.audio
        else:
            print("No active recording")
            return None

def start_recording(recorder):
    recorder.start_recording()

def stop_recording(recorder):
    audio_data = recorder.stop_recording()
    if audio_data is not None:
        wav.write("recording.wav", recorder.fs, audio_data)

def main():
    recorder = Recorder()

    window = tk.Tk()
    window.title("Audio Recorder")
    window.geometry("300x200")  # Set the size of the window

    start_button = tk.Button(window, text="Bắt đầu", command=lambda: start_recording(recorder), height = 2, width = 20)
    start_button.pack(pady=10)  # Add some vertical padding

    stop_button = tk.Button(window, text="Kết thúc", command=lambda: stop_recording(recorder), height = 2, width = 20)
    stop_button.pack(pady=10)  # Add some vertical padding

    window.mainloop()

if __name__ == "__main__":
    main()
