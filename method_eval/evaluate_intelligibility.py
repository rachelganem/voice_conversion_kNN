import os
import csv
import whisper
import torchaudio
import numpy as np
from jiwer import wer, cer
from tqdm import tqdm

import whisper.audio


def load_audio_torchaudio(path: str, sr: int = 16000) -> np.ndarray:
    waveform, sample_rate = torchaudio.load(path)
    if sample_rate != sr:
        waveform = torchaudio.functional.resample(waveform, sample_rate, sr)
    waveform = waveform.mean(dim=0)
    return waveform.numpy()


whisper.audio.load_audio = load_audio_torchaudio


def load_transcripts(csv_path):
    ref_dict = {}
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ref_dict[row['utt_id']] = row['transcript'].strip().upper()
    return ref_dict


class IntelligibilityEvaluator:
    def __init__(self, transcript_csv, whisper_model="base"):
        self.transcripts = load_transcripts(transcript_csv)
        self.model = whisper.load_model(whisper_model)

    def transcribe(self, audio_path):
        result = self.model.transcribe(audio_path, language='en', fp16=False)
        return result['text'].strip().upper()

    def evaluate(self, audio_dir):
        wer_list, cer_list = [], []

        for filename in tqdm(os.listdir(audio_dir)):
            if not filename.endswith(".wav"):
                continue
            utt_id = filename.replace(".wav", "").split("_to_")[0]
            audio_path = os.path.join(audio_dir, filename)

            if utt_id not in self.transcripts:
                print(f"Transcript missing for: {utt_id}")
                continue

            reference = self.transcripts[utt_id]
            hypothesis = self.transcribe(audio_path)

            wer_list.append(wer(reference, hypothesis))
            cer_list.append(cer(reference, hypothesis))

        wer_avg = sum(wer_list) / len(wer_list)
        cer_avg = sum(cer_list) / len(cer_list)

        print(f"Average WER: {wer_avg:.4f}")
        print(f"Average CER: {cer_avg:.4f}")

        return {
            "WER": wer_avg,
            "CER": cer_avg
        }
