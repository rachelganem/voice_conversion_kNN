import os
import csv
import whisper
import torchaudio
from jiwer import wer, cer
from tqdm import tqdm


def load_transcripts(csv_path):
    """
    Load reference transcripts from a CSV file with 'utt_id,transcript' columns.
    """
    ref_dict = {}
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ref_dict[row['utt_id']] = row['transcript'].strip().upper()
    return ref_dict

def transcribe_with_whisper(model, audio_path):
    """
    Transcribe an audio file using Whisper.
    """
    result = model.transcribe(audio_path, language='en', fp16=False)
    return result['text'].strip().upper()

def eval_intelligibility(audio_dir, transcript_csv):
    model = whisper.load_model("base")
    transcripts = load_transcripts(transcript_csv)
    wer_list, cer_list = [], []

    for filename in tqdm(os.listdir(audio_dir)):
        if not filename.endswith(".wav"):
            continue
        utt_id = filename.replace(".wav", "")
        audio_path = os.path.join(audio_dir, filename)

        if utt_id not in transcripts:
            print(f"Transcript missing for: {utt_id}")
            continue

        reference = transcripts[utt_id]
        hypothesis = transcribe_with_whisper(model, audio_path)

        sample_wer = wer(reference, hypothesis)
        sample_cer = cer(reference, hypothesis)

        wer_list.append(sample_wer)
        cer_list.append(sample_cer)

    avg_wer = sum(wer_list) / len(wer_list)
    avg_cer = sum(cer_list) / len(cer_list)
    print("\n--- Intelligibility Evaluation Summary ---")
    print(f"Average WER: {avg_wer:.4f}")
    print(f"Average CER: {avg_cer:.4f}")
    return avg_wer, avg_cer

def eval()


if __name__ == "__main__":


    eval_intelligibility(args.audio_dir, args.transcript_csv, args.output_csv)
