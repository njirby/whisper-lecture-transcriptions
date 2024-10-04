import os
import torch
import librosa
import soundfile as sf
import pandas as pd
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from tqdm import tqdm

# Device setup for GPU or CPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load the model and processor
model_id = "openai/whisper-large-v3-turbo"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
).to(device)

processor = AutoProcessor.from_pretrained(model_id)

# Initialize the ASR pipeline
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

def find_mp3_files(parent_dir):
    """
    Recursively find all .mp3 files in a directory and its subdirectories.
    """
    mp3_files = []
    for root, _, files in os.walk(parent_dir):
        for file in files:
            if file.endswith(".wav"):
                mp3_files.append(os.path.join(root, file))
    return mp3_files

def extract_metadata_from_path(file_path):
    """
    Extract lesson name and lesson module from a file path.
    """
    parts = file_path.split(os.sep)
    lesson_module = parts[-2]  # The directory just before the file
    lesson_name = os.path.splitext(os.path.basename(file_path))[0]  # File name without extension
    return lesson_module, lesson_name

def transcribe_and_save(parent_dir, output_csv_path):
    """
    Transcribe all mp3 files in a directory structure and save results to a CSV file.
    """
    # Find all .mp3 files
    mp3_files = find_mp3_files(parent_dir)
    
    # Prepare an empty list to store results
    records = []
    num = 0
    # Wrap transcription loop in a tqdm progress bar
    for file_path in tqdm(mp3_files, desc="Transcribing audio files"):
        # Extract lesson metadata
        lesson_module, lesson_name = extract_metadata_from_path(file_path)
        
        try:
            # Load and process the audio file
            audio, sample_rate = librosa.load(file_path, sr=16000)
            sf.write("temp_.wav", audio, sample_rate)

            # Transcribe the audio file
            result = pipe("temp_.wav", return_timestamps=True)
            transcription_text = ' '.join([chunk['text'] for chunk in result['chunks']])

            # Append the results to the records list
            records.append({
                "lesson_module": lesson_module,
                "lesson_name": lesson_name,
                "transcription": transcription_text
            })
            
            num += 1

        except Exception as e:
            print(f"Error transcribing {file_path}: {e}")
    
    # Convert records to a DataFrame
    df = pd.DataFrame(records)
    
    # Save DataFrame to a CSV file
    df.to_parquet(f'{parent_dir}/{output_csv_path}', index=False)
    print(f"Saved transcriptions to {output_csv_path}")

# Define the directory and output CSV path
parent_dir = "/data/nate/school/lectures"
output_csv_path = "transcriptions.parquet"

# Run the transcription and saving process
transcribe_and_save(parent_dir, output_csv_path)
