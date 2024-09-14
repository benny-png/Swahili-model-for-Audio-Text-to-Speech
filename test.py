import torch
import numpy as np
import scipy.io.wavfile
from transformers import AutoTokenizer
from vits_model import VitsModel  # Assuming VitsModel is the class for this TTS model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "Benjamin-png/swahili-mms-tts-finetuned"
text = "Habari, karibu kwenye mfumo wetu wa kusikiliza kwa Kiswahili."
audio_file_path = "swahili_speech.wav"

# Load model and tokenizer dynamically based on the provided model name
model = VitsModel.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 1: Tokenize the input text
inputs = tokenizer(text, return_tensors="pt").to(device)

# Step 2: Generate waveform
with torch.no_grad():
    output = model(**inputs).waveform

# Step 3: Convert PyTorch tensor to NumPy array
output_np = output.squeeze().cpu().numpy()

# Step 4: Write to WAV file
scipy.io.wavfile.write(audio_file_path, rate=model.config.sampling_rate, data=output_np)
