from evaluate_intelligibility import IntelligibilityEvaluator
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
transcript_csv = os.path.join(SCRIPT_DIR, "converted", "transcripts.csv")
converted_dir = os.path.join(SCRIPT_DIR, "converted")

if __name__ == '__main__':
    eval_intelligibility = IntelligibilityEvaluator(transcript_csv)

    print("calc intelligibility NORMAL:")
    eval_intelligibility.evaluate(os.path.join(converted_dir, "normal"))

    print("calc intelligibility PREMATCHED:")
    eval_intelligibility.evaluate(os.path.join(converted_dir, "prematched"))
