import argparse
import csv
import os
import random
from collections import defaultdict
from glob import glob

from run_pipline import create_knn_vc_instance, run_full_pipeline_on_existing_knnvc


def build_evaluation_set(test_path, output_csv, num_speakers, num_utterance_per_speaker, seed=123):
    """
    Builds an evaluation CSV file with num_utterance_per_speaker utterances
    sampled from num_speakers. Paths are saved relative to `test_path`.
    """
    print(f"Building evaluation set from: {test_path}")
    random.seed(seed)
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
    assert len(eligible_speakers) >= num_speakers, "Not enough eligible speakers."

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
    print(f"Saved {len(sampled_utterances)} utterances to {output_csv}")


def extract_eval_transcripts(eval_csv_path, data_path, transcript_path):
    transcript_lines = []
    print(f"Extracting transcripts using eval set: {eval_csv_path}")
    with open(eval_csv_path, "r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            audio_path = row["path"]
            utt_id = os.path.splitext(os.path.basename(audio_path))[0]
            trans_file = audio_path.replace(".flac", ".trans.txt").replace(utt_id, utt_id.split("-")[0] + "-" +
                                                                           utt_id.split("-")[1])
            trans_path = os.path.join(data_path, trans_file)
            print(f"Extracting transcript from {trans_path}")
            if os.path.exists(trans_path):
                with open(trans_path, "r", encoding="utf-8") as tf:
                    for line in tf:
                        if line.startswith(utt_id):
                            transcript_lines.append({"utt_id": utt_id, "transcript": line.strip().split(" ", 1)[1]})
                            break
            else:
                print(f"Warning: {trans_path} not found")

    # Save filtered transcripts
    with open(transcript_path, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=["utt_id", "transcript"])
        writer.writeheader()
        writer.writerows(transcript_lines)

    print(f"Saved {len(transcript_lines)} transcripts to {transcript_path}")


def load_eval_set(csv_path):
    speaker_to_utterances = defaultdict(list)
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            speaker_to_utterances[row["speaker"]].append(row["path"])
    return speaker_to_utterances


def run_conversions(output_dir, speaker_to_utterances, all_speakers, data_root, knn_vc, custom, prematched, k):
    custom_dir = "custom" if custom else "original"
    prematched_dir = "prematched" if prematched else "normal"
    output_path = os.path.join(output_dir, custom_dir, prematched_dir)
    os.makedirs(output_path, exist_ok=True)

    for source_speaker, source_utterances in speaker_to_utterances.items():
        for source_path in source_utterances:
            for target_speaker in all_speakers:
                if target_speaker == source_speaker:
                    continue
                # retrieve target utterances
                target_paths = speaker_to_utterances[target_speaker]
                src_wav = os.path.join(data_root, source_path)
                ref_wavs = [os.path.join(data_root, target_path) for target_path in target_paths]
                utt_id = os.path.splitext(os.path.basename(source_path))[0]
                tgt_id = target_speaker
                out_path = os.path.join(output_path, f"{utt_id}_to_{tgt_id}.wav")

                print(f"prematched={prematched} {utt_id}_to_{tgt_id}.wav")

                run_full_pipeline_on_existing_knnvc(knn_vc, src_wav_path=src_wav, ref_wav_paths=ref_wavs,
                                                    output_path=out_path, k=k)


def convert_eval_set(output_dir, csv_path, data_root, device="cuda", run_all=True, use_custom_path=True,
                     prematched=True, k=4):
    print(f"Starting all conversions based on: {csv_path}")
    speaker_to_utterances = load_eval_set(csv_path)
    all_speakers = list(speaker_to_utterances.keys())
    if not run_all:
        knn_vc = create_knn_vc_instance(prematched, device, use_custom_path)
        run_conversions(output_dir, speaker_to_utterances, all_speakers, data_root, knn_vc, use_custom_path, prematched,
                        k)
        print("Done")
        return

    print("Running all Configurations")
    for custom in [True, False]:
        for prematch in [True, False]:
            print(f"========Running Custom={custom}, Prematched={prematch} conversion...=======")
            knn_vc = create_knn_vc_instance(prematch, device, custom)
            run_conversions(output_dir, speaker_to_utterances, all_speakers, data_root, knn_vc, custom, prematch, k)

    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, required=True, help='Path to the test-clean directory')
    parser.add_argument('--csv_path', type=str, default='eval_set.csv', help='Where to save/load eval_set.csv')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for inference')
    parser.add_argument('--prematched', action='store_true', help='Run only prematched version')
    parser.add_argument('--use_custom_path', action='store_true', help='Use local vocoder weights')
    parser.add_argument('--run_all', action='store_true',
                        help='Run both custom and original on prematched and non-prematched ')
    parser.add_argument('--num_speakers', type=int, default=40, help='Number of speakers to sample')
    parser.add_argument('--num_utterances', type=int, default=5, help='Number of utterances per speaker')
    parser.add_argument('--seed', type=int, default=123, help='Random seed for reproducibility')
    parser.add_argument('--k', type=int, default=4, help='Number of nearest neighbors to use during matching')
    parser.add_argument('--output_dir', type=str, default='converted', help='Base output directory')

    args = parser.parse_args()

    build_evaluation_set(test_path=args.test_path, output_csv=args.csv_path, num_speakers=args.num_speakers,
                         num_utterance_per_speaker=args.num_utterances, seed=args.seed)
    extract_eval_transcripts(eval_csv_path=args.csv_path, data_path=args.test_path, transcript_path=os.path.join(args.output_dir, "transcripts.csv"))
    convert_eval_set(output_dir=args.output_dir, csv_path=args.csv_path, data_root=args.test_path, device=args.device,
                     run_all=args.run_all, use_custom_path=args.use_custom_path, prematched=args.prematched, k=args.k)
