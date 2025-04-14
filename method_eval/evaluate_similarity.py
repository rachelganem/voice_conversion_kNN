import os
import csv
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve
from speechbrain.inference.speaker import SpeakerRecognition

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


class SpeakerSimilarityEvaluator:
    """
    Computes speaker similarity of converted utterances using ECAPA-TDNN embeddings
    and evaluates Equal Error Rate (EER) based on positive (same speaker) and
    negative (different speaker) trials.
    """

    # def __init__(self, eval_csv, raw_root, device="cuda"):
    #     """
    #     Args:
    #         eval_csv (str): CSV file mapping speaker -> real utterance paths
    #         raw_root (str): Path to root folder of original test-clean data
    #         device (str): Device to load model and audio on (e.g., "cuda" or "cpu")
    #     """
    #     self.eval_csv = eval_csv
    #     self.raw_root = raw_root
    #     self.device = device
    #     self.model = self._load_verification_model()
    #     self.eval_dict = self._load_eval_set()
    #
    # def _load_verification_model(self):
    #     """Load ECAPA-TDNN speaker verification model from SpeechBrain"""
    #     print("Loading speaker verification model...")
    #     return SpeakerRecognition.from_hparams(
    #         source="speechbrain/spkrec-ecapa-voxceleb",
    #         savedir="pretrained_models/spkrec-ecapa-voxceleb",
    #         run_opts={"device": self.device},
    #         fetch_opts={"use_symlinks": False, "use_tar": True}
    #     )

    def _load_eval_set(self):
        """Load evaluation CSV: speaker → list of real utterances"""
        speaker_utterances = {}
        with open(self.eval_csv, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                speaker = row['speaker']
                speaker_utterances.setdefault(speaker, []).append(row['path'])
        return speaker_utterances

    # def _compute_embedding(self, wav_path, relative_to_root=False):
    #     """
    #     Compute ECAPA-TDNN embedding from .wav file
    #     Args:
    #         wav_path (str): Path to .wav file (absolute or relative)
    #         relative_to_root (bool): If True, resolves path relative to raw_root
    #     Returns:
    #         np.ndarray: 1D speaker embedding
    #     """
    #     full_path = os.path.join(self.raw_root, wav_path) if relative_to_root else wav_path
    #     signal, sr = torchaudio.load(full_path)
    #     if sr != 16000:
    #         signal = torchaudio.functional.resample(signal, sr, 16000)
    #     signal = signal.to(self.device)
    #     embedding = self.model.encode_batch(signal).squeeze(0).squeeze(0).cpu().numpy()
    #     return embedding
    #
    # @staticmethod
    # def _cosine_similarity(a, b):
    #     """Compute cosine similarity between two vectors"""
    #     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    #
    # @staticmethod
    # def _compute_eer(scores, labels):
    #     """Compute EER (Equal Error Rate) and its corresponding threshold"""
    #     fpr, tpr, thresholds = roc_curve(labels, scores)
    #     fnr = 1 - tpr
    #     eer_idx = np.nanargmin(np.abs(fnr - fpr))
    #     return fpr[eer_idx], thresholds[eer_idx]
    #
    # def _generate_trials(self, converted_pairs):
    #     """
    #     For each converted utterance, generate one positive (same speaker)
    #     and one negative (different speaker) similarity score.
    #
    #     Args:
    #         converted_pairs (List): List of (converted_path, source_spk, target_spk)
    #     Returns:
    #         scores (np.ndarray): Cosine similarity scores
    #         labels (np.ndarray): 1 = same speaker, 0 = different speaker
    #     """
    #     scores, labels = [], []
    #     speakers = list(self.eval_dict.keys())
    #
    #     for converted_path, _, tgt_spk in tqdm(converted_pairs):
    #         emb_converted = self._compute_embedding(converted_path)
    #
    #         # Positive trial
    #         pos_utt = np.random.choice(self.eval_dict[tgt_spk])
    #         emb_pos = self._compute_embedding(pos_utt, relative_to_root=True)
    #         scores.append(self._cosine_similarity(emb_converted, emb_pos))
    #         labels.append(1)
    #
    #         # Negative trial
    #         neg_spk = np.random.choice([s for s in speakers if s != tgt_spk])
    #         neg_utt = np.random.choice(self.eval_dict[neg_spk])
    #         emb_neg = self._compute_embedding(neg_utt, relative_to_root=True)
    #         scores.append(self._cosine_similarity(emb_converted, emb_neg))
    #         labels.append(0)
    #
    #     return np.array(scores), np.array(labels)
    #
    # def evaluate(self, converted_dir):
    #     """
    #     Run full evaluation pipeline for a directory of converted utterances.
    #
    #     Args:
    #         converted_dir (str): Directory containing <source>_to_<target>.wav files
    #
    #     Returns:
    #         eer (float): Equal Error Rate
    #         threshold (float): Cosine similarity threshold for EER
    #     """
    #     print(f"Evaluating similarity for: {converted_dir}")
    #     converted_pairs = self._load_converted_pairs(converted_dir)
    #     scores, labels = self._generate_trials(converted_pairs)
    #     eer, threshold = self._compute_eer(scores, labels)
    #
    #     print("\n--- Speaker Similarity Evaluation ---")
    #     print(f"EER (Equal Error Rate): {eer:.4f}")
    #     print(f"Threshold: {threshold:.4f}")
    #     return eer, threshold
    #
    @staticmethod
    def _load_converted_pairs(converted_dir):
        """Parse converted file paths of format <src>_to_<target>.wav"""
        pairs = []
        for file in os.listdir(converted_dir):
            if not file.endswith(".wav"):
                continue
            src, tgt = file.replace(".wav", "").split("_to_")
            pairs.append((os.path.join(converted_dir, file), src, tgt))
        return pairs
