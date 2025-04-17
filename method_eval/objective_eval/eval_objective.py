import argparse
import os
from evaluate_intelligibility import IntelligibilityEvaluator
from evaluate_speaker_similarity import SpeakerSimilarityEvaluator


def ensure_output_dir(output_path):
    """Ensure the directory for the output file exists."""
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)


def run_intelligibility(transcript_csv, converted_path, whisper_model, output_path):
    evaluator = IntelligibilityEvaluator(transcript_csv, whisper_model=whisper_model)
    print(f"Calculating intelligibility for: {converted_path}")
    results = evaluator.evaluate(converted_path)

    with open(output_path, "a", encoding="utf-8") as f:
        f.write(f"[Intelligibility] {os.path.basename(converted_path)},WER={results['WER']:.4f},CER={results['CER']:.4f}\n")
    print(f"Results saved to {output_path}")

def run_speaker_similarity(data_path, converted_path, output_path, device):

    evaluator = SpeakerSimilarityEvaluator(data_path, device=device)
    print(f"Calculating speaker similarity for: {converted_path}")
    results = evaluator.evaluate(converted_path)

    with open(output_path, "a", encoding="utf-8") as f:
        f.write(f"[Similarity] {os.path.basename(converted_path)},Similarity={results['similarity']:.4f}\n")
    print(f"Results saved to {output_path}")

def main():
    print("start")
    parser = argparse.ArgumentParser(description="Run objective evaluations (intelligibility / speaker similarity)")
    parser.add_argument(
        "--eval", nargs="+", choices=["intelligibility", "speaker_similarity"], required=True,
        help="Evaluation types to run (you can pass multiple)"
    )
    parser.add_argument('--test_path', type=str, required=True, help='Path to the test-clean directory')
    parser.add_argument("--converted_path", required=True, help="Full path to the folder with converted audio files")
    parser.add_argument("--output_path", required=True, help="Where to save results")
    # Only required for intelligibility
    parser.add_argument("--transcript_csv", help="Path to transcript CSV (required for intelligibility)")
    parser.add_argument("--whisper_model", default="base", help="Whisper model to use")
    parser.add_argument("--device", default="cpu", help="Device to run models on")

    args = parser.parse_args()

    if not os.path.isdir(args.converted_path):
        print(f"Directory not found: {args.converted_path}")
        return

    ensure_output_dir(args.output_path)

    if "intelligibility" in args.eval:
        if not args.transcript_csv:
            print("Error: --transcript_csv is required for intelligibility evaluation.")
            return
        run_intelligibility(args.transcript_csv, args.converted_path, args.whisper_model, args.output_path)

    if "speaker_similarity" in args.eval:
        run_speaker_similarity(args.test_path, args.converted_path, args.output_path, args.device)

if __name__ == '__main__':
    main()
