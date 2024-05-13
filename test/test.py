import speech_recognition as sr
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import io
import torch
from pydub import AudioSegment


tokenizer = Wav2Vec2Processor.from_pretrained('JustAFool/wav2vec2-vi-300-2')
model = Wav2Vec2ForCTC.from_pretrained('JustAFool/wav2vec2-vi-300-2')
r = sr.Recognizer()

with sr.Microphone(sample_rate=16000) as source:
    print('Say something now!')
    while True:
        try:
            audio = r.listen(source)
            data = io.BytesIO(audio.get_wav_data())
            clip = AudioSegment.from_file(data)
            x = torch.FloatTensor(clip.get_array_of_samples())

            inputs = tokenizer(x, sampling_rate=16000, return_tensors='pt', padding='longest').input_values
            logits = model(inputs).logits
            tokens = torch.argmax(logits, axis=-1)
            text = tokenizer.batch_decode(tokens)
            print('You said: ', str(text).lower())
        except KeyboardInterrupt:
            print("Stopped by user!")
        except Exception as e:
            print("An error occurred: ", str(e))