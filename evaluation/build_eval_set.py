import os
import random
from glob import glob
import csv
from collections import defaultdict

csv_path = "eval_set.csv"
output_prematched_path = "converted/prematched"
output_normal_path = "converted/normal"


def build_evaluation_set(test_path, output_csv, num_speakers=40, num_utterance_per_speaker=5):
    """
    Builds an evaluation CSV file with num_utterance_per_speaker utterances
    sampled from num_speakers. Paths are saved relative to `test_path`.
    """
    print("Building Evaluation Set...")
    speaker_to_files = defaultdict(list)

    # Step 1: Find all .flac files grouped by speaker
    for speaker_id in os.listdir(test_path):
        speaker_path = os.path.join(test_path, speaker_id)
        if not os.path.isdir(speaker_path):
            continue
        for chapter_id in os.listdir(speaker_path):
            chapter_path = os.path.join(speaker_path, chapter_id)
            if not os.path.isdir(chapter_path):
                continue
            flac_files = glob(os.path.join(chapter_path, "*.flac"))
            speaker_to_files[speaker_id].extend(flac_files)

    # Step 2: Filter speakers with enough files
    eligible_speakers = [s for s in speaker_to_files if len(speaker_to_files[s]) >= num_utterance_per_speaker]
    assert len(eligible_speakers) >= num_speakers, "❌ Not enough eligible speakers."

    # Step 3: Sample speakers and utterances
    selected_speakers = random.sample(eligible_speakers, num_speakers)
    sampled_utterances = []

    for speaker in selected_speakers:
        chosen = random.sample(speaker_to_files[speaker], num_utterance_per_speaker)
        for path in chosen:
            relative_path = os.path.relpath(path, start=test_path)
            sampled_utterances.append({
                "speaker": speaker,
                "path": relative_path
            })

    # Step 4: Save to CSV
    with open(output_csv, "w", newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["speaker", "path"])
        writer.writeheader()
        writer.writerows(sampled_utterances)


def extract_eval_transcripts(eval_csv_path=csv_path, output_path="converted/transcripts.csv"):
    transcript_lines = []

    with open(eval_csv_path, "r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            audio_path = row["path"]
            utt_id = os.path.splitext(os.path.basename(audio_path))[0]
            trans_file = audio_path.replace(".flac", ".trans.txt").replace(utt_id, utt_id.split("-")[0] + "-" +
                                                                           utt_id.split("-")[1])

            if os.path.exists(trans_file):
                with open(trans_file, "r", encoding="utf-8") as tf:
                    for line in tf:
                        if line.startswith(utt_id):
                            transcript_lines.append({"utt_id": utt_id, "transcript": line.strip().split(" ", 1)[1]})
                            break
            else:
                print(f"Warning: {trans_file} not found")

    # Save filtered transcripts
    with open(output_path, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=["utt_id", "transcript"])
        writer.writeheader()
        writer.writerows(transcript_lines)

    print(f"Saved {len(transcript_lines)} transcripts to {output_path}")


def load_eval_set(csv_path):
    speaker_to_utterances = defaultdict(list)
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            speaker_to_utterances[row["speaker"]].append(row["path"])
    return speaker_to_utterances


def run_all_conversions(pre_matched=True, csv_path=csv_path):
    output_path = output_prematched_path if pre_matched else output_normal_path
    speaker_to_utterances = load_eval_set(csv_path)
    all_speakers = list(speaker_to_utterances.keys())

    for source_speaker, source_utterances in speaker_to_utterances.items():
        for source_path in source_utterances:
            for target_speaker in all_speakers:
                if target_speaker == source_speaker:
                    continue
                target_path = random.choice(speaker_to_utterances[target_speaker])
                run_pipeline(source_path, target_path, output_path)  # todo: import this function


if __name__ == "__main__":
    test_folder = "../data/dataset/raw/test-clean/test-clean"
    build_evaluation_set(test_folder)
    # run_all_conversions(pre_matched=True)
    # run_all_conversions(pre_matched=False)
    extract_eval_transcripts()
