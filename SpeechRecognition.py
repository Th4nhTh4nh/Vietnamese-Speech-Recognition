import tkinter as tk
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import torch
import librosa
import os
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from pydub import AudioSegment
from tkinter import filedialog

class Recorder:
    def __init__(self):
        self.recording = False
        self.fs = 44100
        self.audio = []
        self.tokenizer = Wav2Vec2Processor.from_pretrained('JustAFool/wav2vec2-vi-300-2')
        self.model = Wav2Vec2ForCTC.from_pretrained('JustAFool/wav2vec2-vi-300-2')
        
    def start_recording(self):
        self.recording = True
        self.audio = []
        sd.default.samplerate = self.fs
        sd.default.channels = 1
        self.stream = sd.InputStream(callback=self.callback)
        self.stream.start()

    def callback(self, indata, frames, time, status):
        if self.recording:
            self.audio.append(indata.copy())

    def stop_recording(self):
        if self.recording:
            self.recording = False
            self.stream.stop()
            self.stream.close()
            audio_array = np.concatenate(self.audio, axis=0)
            wav.write('./audio/output.wav', self.fs, audio_array)

            file = './audio/output.wav'
            audio, rate = librosa.load(file)
            audio = librosa.resample(audio, orig_sr=rate, target_sr=16000)
            input_audio = torch.tensor(audio)
            input_values = self.tokenizer(input_audio, return_tensors='pt', padding='longest').input_values
            logits = self.model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            predicted_sentence = self.tokenizer.batch_decode(predicted_ids)
            #print(predicted_sentence)
            return predicted_sentence

        
        
recorder = Recorder()

def on_button_click():
    if not recorder.recording:
        recorder.start_recording()
        record_button.config(text='Stop')
    else:
        result = recorder.stop_recording()
        record_button.config(text='Record')
        text.config(state='normal')
        text.insert(tk.END, ''.join(result))
        text.config(state='disabled')

def clear_text():
    text.config(state='normal')
    text.delete('1.0', tk.END)


def import_audio():
    filename = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3 *.flac")])
    # Tiếp tục xử lý file âm thanh tại đây
    audio, rate = librosa.load(filename)
    audio = librosa.resample(audio, orig_sr=rate, target_sr=16000)
    input_audio = torch.tensor(audio)
    input_values = recorder.tokenizer(input_audio, return_tensors='pt', padding='longest').input_values
    logits = recorder.model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    predicted_sentence = recorder.tokenizer.batch_decode(predicted_ids)
    text.config(state='normal')
    text.insert(tk.END, ''.join(predicted_sentence))
    text.config(state='disabled')


def export_text():
    content = text.get('1.0', tk.END)
    if not content.strip():
        tk.messagebox.showinfo("Empty content", "The text is empty.")
        return
    file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
    if file_path:
        try:
            with open(file_path, "w", encoding='utf-8') as file:
                file.write(content)
                tk.messagebox.showinfo("Info", "File saved successfully.")
        except Exception as e:
            tk.messagebox.showinfo("Error", f"An error occurred while saving the file: {str(e)}")


root = tk.Tk()

root.title("Vietnamese Speech Recognition")
root.geometry("250x190")

frame = tk.Frame(root, bd=2, relief='groove')  # Tạo border tại đây
frame.pack()

text = tk.Text(frame, height=5, width=50)
text.config(state='disabled')
text.pack(pady=10)

button_frame = tk.Frame(root)
button_frame.pack(pady=10)

record_button = tk.Button(button_frame, text='Record Audio', command=on_button_click, width=12)
record_button.grid(row=0, column=0, padx=5)

clear_button = tk.Button(button_frame, text='Clear Text',command=clear_text, width=12)
clear_button.grid(row=1, column=0, padx=5, pady=5)

choose_button = tk.Button(button_frame, text='Choose File', command=import_audio, width=12)
choose_button.grid(row=0, column=1, padx=5)

export_button = tk.Button(button_frame, text='Export Text', command=export_text, width=12)
export_button.grid(row=1, column=1, padx=5, pady=5)

root.mainloop()
