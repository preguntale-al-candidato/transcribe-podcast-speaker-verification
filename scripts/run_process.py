import pandas as pd
import os
import yt_dlp
import yaml
import argparse
import whisperx
import torch
import json
import time
import shutil

from tqdm import tqdm
from pathlib import Path
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment
from scipy.spatial.distance import cdist

reference = {
    "MILEI PRESIDENTE": "milei",
    "El Peluca Milei": "milei",
    "Javier Milei": "milei",
    "Sergio Massa": "massa",
    "Patricia Bullrich": "bullrich",
}

def download_audio(URL, output_dir, audio_fname:str = 'audio'):
    """
    Download the audio from a YouTube video as a WAV file.
    Args:
        URL (str): The URL of the YouTube video.
        output_dir (str): The directory to save the output file in.
        audio_filename (str): The filename to save the output file as.
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        'outtmpl': os.path.join(output_dir, audio_fname),
        'format': 'm4a/bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }]
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download(URL)
        
def save_dict_to_disk(data_dict, data, reference, save_dir, filename:str = None):
    """
    Save a dictionary to disk as a JSON file.
    Args:
        data_dict (dict): The dictionary to save.
        data (namedtuple): A named tuple representing a row from a dataframe.
        reference (dict): A dictionary mapping channel names to some values.
        filename (str): The filename to save the output file as (without extension).
    """
    output_dir = f"{save_dir}/{reference[data.channel_name]}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filepath = os.path.join(output_dir, f"{filename}.json")
    
    # convert the named tuple to a dictionary
    row_dict = data._asdict()
    
    # merge the two dictionaries
    combined_dict = {**data_dict, **row_dict}
    
    with open(filepath, 'w') as f:
        json.dump(combined_dict, f)
    
    print(f"dict saved to {filepath}")
    
    
def main():
    parser = argparse.ArgumentParser(description='process and transcribe YouTube videos.')
    parser.add_argument('--hf_token', help='Hugging Face Token')
    parser.add_argument('--cdist_threshold', type=float, default=0.5, help='Threshold for cdist (default: 0.5)')
    parser.add_argument('--data_dir', default='../data/youtube_data', help='Directory containing the data file (default: ../data/youtube_data)')
    parser.add_argument('--ref_audio_dir', default='../data/reference_audio', help='Reference audio directory (default: ../data/reference_audio)')
    parser.add_argument('--temp_dir', default='../data/temp_audio', help='Temporary directory for intermediate files (default: ../data/temp_audio)')
    parser.add_argument('--output_dir', default='../output', help='Output directory for saving results (default: ../output)')
    parser.add_argument('--device', default="cuda", help='Device for processing (default: cuda)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing (default: 32)')
    parser.add_argument('--compute_type', default="float16", help='Compute type for processing (default: float16)')
    args = parser.parse_args()

    # assigning variables from argparse
    hf_token = args.hf_token
    cdist_threshold = args.cdist_threshold
    data_dir = args.data_dir
    ref_audio_dir = args.ref_audio_dir
    temp_dir = args.temp_dir
    output_dir = args.output_dir
    target_audio = f"{args.temp_dir}/audio.wav"
    input_data = f'{args.data_dir}/data.csv'
    device = args.device
    batch_size = args.batch_size
    compute_type = args.compute_type
    
    # load whisper model
    model = whisperx.load_model("large-v2", device, compute_type=compute_type)

    # load embeddings model
    embeddings_model = PretrainedSpeakerEmbedding(
        "speechbrain/spkrec-ecapa-voxceleb", device=device)
    
    data = pd.read_csv(input_data)
    for each in tqdm(data.itertuples(),total=data.shape[0]):
        
        # check if file was already processed
        print('\n---------\n')
        if os.path.exists(f'{output_dir}/{reference[each.channel_name]}/{each.id}.json'):
            print('\033[1m' + f'skipping: {each.title}' + '\033[0m')
            continue
        else:
            print('\033[1m' + f'processing now: {each.title}' + '\033[0m')
        
        # download audio from youtube url
        download_audio(each.url, temp_dir)

        # transcribe and perform speaker diarization
        audio = whisperx.load_audio(target_audio)
        result = model.transcribe(audio, batch_size=batch_size, language='es')

        # align whisper output
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

        # assign speaker labels
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
        diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)

        # start audio pipeline
        audio_pipe = Audio(sample_rate=16000, mono="downmix")
        reference_wav_fname = f'{ref_audio_dir}/audio_{reference[each.channel_name]}.wav'

        # extract embeddings for a speaker speaking between t=0 and t=10s
        segment_reference = Segment(1., 10.)
        waveform_reference, sample_rate = audio_pipe.crop(reference_wav_fname, segment_reference)
        embedding_reference = embeddings_model(waveform_reference[None])

        # add cosine dist to each segment in the results
        for segment in result['segments']:
            # extract embedding for a speaker speaking between t=Xs and t=Ys
            speaker_target = Segment(segment['start'], segment['end'])
            waveform_target, sample_rate = audio_pipe.crop(target_audio, speaker_target)
            embedding_target = embeddings_model(waveform_target[None])

            # compare embeddings using "cosine" distance
            distance = cdist(embedding_reference, embedding_target, metric="cosine")
            segment['cosine_dist'] = distance[0][0]

            # save back the info to the dict
            segment['is_candidate'] = True if distance[0][0] <= cdist_threshold else False

        # save result dictionary to local disk
        save_dict_to_disk(result, each, reference, output_dir, filename=each.id)
        
        # delete temp folders
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()