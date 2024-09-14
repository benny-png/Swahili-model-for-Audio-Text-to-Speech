

# Swahili MMS TTS - Finetuned Model

This is a fine-tuned version of the Facebook MMS (Massively Multilingual Speech) model for Swahili Text-to-Speech (TTS). The model was fine-tuned to improve Swahili pronunciation and performance using custom audio datasets.

## Model Details

- **Model Name**: Swahili MMS TTS - Finetuned
- **Languages Supported**: Swahili
- **Base Model**: Facebook MMS
- **Use Case**: Text-to-Speech for Swahili language, suitable for generating high-quality speech from text.

## Training Details

The fine-tuning process was done using a custom dataset of Swahili voice samples to improve the fluency and accuracy of the original MMS model in Swahili. This resulted in enhanced pronunciation and natural-sounding speech for Swahili.

You can check out the code and process used in the fine-tuning by visiting the [GitHub repository](https://github.com/benny-png/Swahili-model-for-Audio-Text-to-Speech).

## How to Use

You can load and use the model directly from the Hugging Face model hub using either the `pipeline` API or by manually downloading the model and tokenizer.


### 1. Download and Run the Model Directly

You can also download the model and tokenizer manually and run the text-to-speech pipeline without the Hugging Face `pipeline` helper. Here's how:

```python
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
```


### 2. Using the `pipeline` API

```python
from transformers import pipeline

# Load the fine-tuned model
tts = pipeline("text-to-speech", model="Benjamin-png/swahili-mms-tts-finetuned")

# Generate speech from text
speech = tts("Habari, karibu kwenye mfumo wetu wa kusikiliza kwa Kiswahili.")
```



### Saving and Playing the Audio

To save and play the audio, you can use the same methods mentioned above:

#### Saving the Audio

```python
import soundfile as sf

# Save the audio as a WAV file
sf.write("swahili_speech.wav", output_np, model.config.sampling_rate)
```

#### Playing the Audio

You can play the audio using `pydub`:

```python
from pydub import AudioSegment
from pydub.playback import play

# Load and play the generated audio
audio = AudioSegment.from_wav("swahili_speech.wav")
play(audio)
```

Make sure to install the required libraries:

```bash
pip install torch transformers numpy soundfile scipy pydub
```

## Example Notebook

If you're interested in reproducing the fine-tuning process or using the model for similar purposes, you can check out the Google Colab notebook that outlines the entire process:

- [Google Colab Notebook](https://colab.research.google.com/drive/1dK1a814UqDnXnM5Rz6NBmk-vmhdN9M4f#scrollTo=iG6IrVva27uT)

The notebook includes detailed steps on how to fine-tune the MMS model for Swahili TTS.

## GitHub Repository

For further exploration and code snippets, visit the [GitHub repository](https://github.com/benny-png/Swahili-model-for-Audio-Text-to-Speech) where you’ll find additional scripts, datasets, and instructions for customizing the model.

## License

This project is licensed under the terms of the Apache License 2.0.

