import os
from pathlib import Path
from moviepy.editor import VideoFileClip
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def process_file(mp4_path):
    wav_path = mp4_path.with_suffix('.wav')
    
    # Load video and convert to audio
    with VideoFileClip(str(mp4_path)) as video:
        audio = video.audio
        audio.write_audiofile(str(wav_path), fps=16000, logger=None)
    
    return mp4_path  # Return path for progress tracking

def convert_mp4_to_wav(parent_dir):
    # List all mp4 files
    mp4_files = [Path(root) / file for root, _, files in os.walk(parent_dir) for file in files if file.endswith(".mp4")]
    
    # Create a pool of workers
    with Pool(cpu_count()) as pool:
        # Wrap pool.imap in tqdm for progress tracking
        for _ in tqdm(pool.imap_unordered(process_file, mp4_files), total=len(mp4_files), desc="Converting MP4 to WAV", unit="file"):
            pass  # `tqdm` updates as each file is processed
    
    print("Conversion complete.")

# Example usage:
convert_mp4_to_wav('/data/nate/school/lectures')
