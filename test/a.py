import tkinter as tk
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import io
import torch
from pydub import AudioSegment
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

class Recorder:
    def __init__(self):
        self.recording = False
        self.fs = 44100
        self.audio = []

        # Khởi tạo tokenizer và model
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
            wav.write('output.wav', self.fs, audio_array)

            # Xử lý nhận diện giọng nói sau khi ghi âm
            clip = AudioSegment.from_file('output.wav', format="wav")
            x = torch.FloatTensor(clip.get_array_of_samples())
            inputs = self.tokenizer(x, sampling_rate=16000, return_tensors='pt', padding='longest').input_values
            logits = self.model(inputs).logits
            tokens = torch.argmax(logits, axis=-1)
            text = self.tokenizer.batch_decode(tokens)
            return text

recorder = Recorder()

def on_button_click():
    if not recorder.recording:
        recorder.start_recording()
        button.config(text='Stop')
    else:
        text = recorder.stop_recording()
        button.config(text='Record')
        # Thêm nội dung vào ô thông tin sau khi ghi âm kết thúc
        info_text.config(state='normal')
        info_text.insert(tk.END, 'Ghi âm đã kết thúc. Nội dung được nhận diện: ' + text)
        info_text.config(state='disabled')

def clear_text():
    info_text.config(state='normal')
    info_text.delete('1.0', tk.END)
    info_text.config(state='disabled')

root = tk.Tk()
root.geometry("500x500")  # Thay đổi kích thước cửa sổ tại đây

# Tạo một khung chứa ô thông tin văn bản ở trên
frame = tk.Frame(root, bd=2, relief='groove')  # Tạo border tại đây
frame.pack()

# Tạo một ô thông tin văn bản trong khung
info_text = tk.Text(frame, width=60, height=10)  # Điều chỉnh kích thước tại đây
info_text.pack()
info_text.config(state='disabled')  # Không cho phép người dùng nhập thông tin

# Tạo một khung chứa các nút
button_frame = tk.Frame(root)
button_frame.pack(pady=10)  # Tạo khoảng cách giữa khung nút và ô thông tin

# Đặt nút "Record/Stop" vào khung
button = tk.Button(button_frame, text='Record', command=on_button_click, width=20)  # Điều chỉnh chiều dài tại đây
button.grid(row=0, column=0, padx=5)  # Sử dụng phương thức grid để sắp xếp nút và tạo khoảng cách

# Tạo và đặt nút "Clear" vào khung, bên cạnh nút "Record/Stop"
clear_button = tk.Button(button_frame, text='Clear', command=clear_text, width=20)  # Điều chỉnh chiều dài tại đây
clear_button.grid(row=0, column=1, padx=5)  # Sử dụng phương thức grid để sắp xếp nút và tạo khoảng cách

root.mainloop()
