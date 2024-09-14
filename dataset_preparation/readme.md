
# Data Preparation for TTS Model Training

This repository contains a Python script for preparing a dataset for training Text-to-Speech (TTS) models using audio files and their corresponding text data. The dataset is created from .wav files and a CSV file containing the text annotations. Once prepared, the dataset is uploaded to the Hugging Face Model Hub.

## Prerequisites

To run the data preparation script, you need to have the following packages installed:

```bash
pip install datasets huggingface_hub
```

## Directory Structure

Ensure your project directory has the following structure:

```
project/
│
├── data/
│   ├── audio/
│   │   ├── 001.wav
│   │   ├── 002.wav
│   │   └── ...        # Your .wav files
│   └── text.csv       # Your CSV file with text data
│
└── data_preparation.py  # This script
```

## CSV File Format

The CSV file should have the following columns:

- **file_name**: Name of the .wav file (e.g., "001.wav")
- **text**: Corresponding text for the audio file
- **speaker_id**: ID of the speaker (optional, default is 1)

### Example CSV Content:

```csv
file_name,text,speaker_id
001.wav,This is the first sentence.,1
002.wav,This is the second sentence.,1
```

## Code Overview

### Main Function

The main function `create_and_upload_dataset` creates a dataset from the provided audio files and CSV text data, and uploads it to Hugging Face.

```python
def create_and_upload_dataset(dataset_dir: str, csv_file: str, repo_id: str) -> None:
```

- **Args**:
  - `dataset_dir` (str): Path to the directory containing .wav files.
  - `csv_file` (str): Path to the CSV file containing text data.
  - `repo_id` (str): Hugging Face repository ID for uploading the dataset.

### Supporting Functions

1. **load_text_data**: Loads text data from the CSV file.

    ```python
    def load_text_data(csv_file: str) -> Dict[str, Dict[str, str]]:
    ```

2. **create_dataset_entries**: Creates dataset entries based on available .wav files and text data.

    ```python
    def create_dataset_entries(dataset_dir: str, sample_texts: Dict[str, Dict[str, str]]) -> List[Dict]:
    ```

3. **upload_to_huggingface**: Uploads the dataset to the Hugging Face Model Hub.

    ```python
    def upload_to_huggingface(dataset: Dataset, repo_id: str) -> None:
    ```

### Example Usage

You can use the script as follows:

```python
create_and_upload_dataset("data/audio", "data/text.csv", "my-dataset")
```

### Example Usage in a FastAPI Route

You can also integrate the dataset creation into a FastAPI application:

```python
@app.post("/create_dataset")
async def create_dataset(dataset_dir: str, csv_file: str, repo_id: str):
    create_and_upload_dataset(dataset_dir, csv_file, repo_id)
    return {"message": "Dataset created and uploaded successfully"}
```

