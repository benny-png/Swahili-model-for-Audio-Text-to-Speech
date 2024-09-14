import os
import csv
from typing import List, Dict
from datasets import Dataset, Audio
from huggingface_hub import HfApi, HfFolder, create_repo

def create_and_upload_dataset(dataset_dir: str, csv_file: str, repo_id: str) -> None:
    """
    Create a dataset from audio files and corresponding text data, then upload it to Hugging Face.

    Args:
    dataset_dir (str): Path to the directory containing .wav files
    csv_file (str): Path to the CSV file containing text data
    repo_id (str): Hugging Face repository ID for uploading the dataset

    The CSV file should have the following columns:
    - file_name: Name of the .wav file (e.g., "001.wav")
    - text: Corresponding text for the audio file
    - speaker_id: ID of the speaker (optional, default is 1)

    Example CSV content:
    file_name,text,speaker_id
    001.wav,This is the first sentence.,1
    002.wav,This is the second sentence.,1
    """

    # Load text data from CSV
    sample_texts = load_text_data(csv_file)

    # Create dataset entries
    data = create_dataset_entries(dataset_dir, sample_texts)

    # Create Dataset
    dataset = Dataset.from_list(data)

    # Add audio feature to the dataset
    dataset = dataset.cast_column("audio", Audio())

    # Upload to Hugging Face
    upload_to_huggingface(dataset, repo_id)

def load_text_data(csv_file: str) -> Dict[str, Dict[str, str]]:
    """Load text data from CSV file."""
    sample_texts = {}
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_texts[row['file_name']] = {
                'text': row['text'],
                'speaker_id': row.get('speaker_id', '1')
            }
    return sample_texts

def create_dataset_entries(dataset_dir: str, sample_texts: Dict[str, Dict[str, str]]) -> List[Dict]:
    """Create dataset entries based on available wav files and text data."""
    data = []
    for filename in sorted(os.listdir(dataset_dir)):
        if filename.endswith('.wav'):
            if filename in sample_texts:
                file_path = os.path.join(dataset_dir, filename)
                line_id = f"BI{int(filename.split('.')[0]):04d}"
                text = sample_texts[filename]['text']
                speaker_id = int(sample_texts[filename]['speaker_id'])

                entry = {
                    'line_id': line_id,
                    'audio': file_path,
                    'text': text,
                    'speaker_id': speaker_id
                }
                data.append(entry)
    return data

def upload_to_huggingface(dataset: Dataset, repo_id: str) -> None:
    """Upload the dataset to Hugging Face."""
    hf_token = HfFolder.get_token()
    api = HfApi()

    try:
        create_repo(repo_id=repo_id, repo_type="dataset", token=hf_token)
        print("Repository created successfully.")
    except Exception as e:
        print(f"Repository creation failed or already exists: {e}")

    dataset.push_to_hub(repo_id, token=hf_token)
    print("Dataset uploaded successfully!")


# Example usage normally:
create_and_upload_dataset("data/audio", "data/text.csv", "my-dataset")


# Example usage in a FastAPI route:
# @app.post("/create_dataset")
# async def create_dataset(dataset_dir: str, csv_file: str, repo_id: str):
#     create_and_upload_dataset(dataset_dir, csv_file, repo_id)
#     return {"message": "Dataset created and uploaded successfully"}


