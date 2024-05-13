import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Tokenizer
from datasets import load_dataset
import soundfile as sf
import librosa
from scipy.io import wavfile

dataset = load_dataset("mozilla-foundation/common_voice_11_0", "vi", split="validation")
sample = dataset[30]
audio = sample["audio"]["array"]
sample_rate = sample["audio"]["sampling_rate"]
audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)


token = Wav2Vec2Tokenizer.from_pretrained("JustAFool/wav2vec2-vi-300-2")
model = Wav2Vec2ForCTC.from_pretrained("JustAFool/wav2vec2-vi-300-2")

input_audio = torch.tensor(audio)
input_values = token(input_audio, return_tensors="pt", padding=True, truncation=True).input_values
logits = model(input_values).logits
predicted_ids = torch.argmax(logits, dim=-1)
predicted_sentence = token.batch_decode(predicted_ids)
print("Predicted: ",predicted_sentence)
text = sample["sentence"]
print("origin: ", text)
"""


"""