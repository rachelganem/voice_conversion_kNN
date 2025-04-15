import os

import numpy as np
import torch
from pyannote.audio import Inference
from pyannote.audio import Model
from scipy.spatial.distance import cosine
from sklearn.metrics import roc_curve

from . import eval_utils as ut

# Force offline mode for Hugging Face Hub
os.environ["HF_HUB_OFFLINE"] = "1"


def cosine_similarity(vector1, vector2):
    """
    Calculate cosine similarity between two vectors.

    Parameters:
    - vector1: numpy array, first vector.
    - vector2: numpy array, second vector.

    Returns:
    - similarity: float, cosine similarity between the two vectors.
    """
    return 1 - cosine(vector1, vector2)


class SpeakerSimilarityEvaluator:
    def __init__(self, test_data_path, device="cpu"):
        try:
            model = Model.from_pretrained("pyannote/embedding", strict=False)

            self.inference = Inference(
                model,
                window="whole",
                device=torch.device(device)  # <= correct usage
            )
        except Exception as e:
            raise RuntimeError(f"Could not load offline pyannote model: {e}")

        self.data_path = test_data_path

    def extract_x_vector(self, audio_path):
        return self.inference(audio_path)

    def calculate_EER(self, positive_similarities, negative_similarities):
        similarities = positive_similarities + negative_similarities
        labels = [1] * len(positive_similarities) + [0] * len(negative_similarities)

        # Compute ROC curve (FPR, TPR)
        fpr, tpr, thresholds = roc_curve(labels, similarities)
        # Identify the index where the difference |FPR - (1-TPR)| is minimized.
        eer_index = np.nanargmin(np.abs(fpr - (1 - tpr)))
        return fpr[eer_index]

    def evaluate(self, converted_path):
        # Build a dictionary mapping target speaker IDs to a random reference .flac file from the test dataset.
        speaker_dict = ut.create_speaker_dict(self.data_path)
        positive_scores = []
        negative_scores = []
        count = 0
        # Traverse the converted directory to process each .wav file.
        for root, dirs, files in os.walk(converted_path):
            for file in files:
                count += 1
                if file.endswith(".wav"):
                    # Get the full audio path.
                    converted_audio_path = os.path.join(root, file)
                    # Extract source and target speaker IDs from the filename.
                    source_id, target_id = ut.get_source_and_target_speaker_from_filename(file)

                    # Extract embeddings: the converted embedding, the reference (positive), and a negative sample.
                    converted_target_vector = self.extract_x_vector(converted_audio_path)
                    positive_target_vect = self.extract_x_vector(speaker_dict[target_id])
                    negative_sample_path = ut.get_random_negative_sample(speaker_dict, source_id, target_id)
                    negative_target_vect = self.extract_x_vector(negative_sample_path)

                    # Compute cosine similarities for positive and negative pairs.
                    positive_scores.append(cosine_similarity(positive_target_vect, converted_target_vector))
                    negative_scores.append(cosine_similarity(negative_target_vect, converted_target_vector))
                    print(f"count = {count}")
                    print(f"positive_scores = {positive_scores[count - 1]}")
                    print(f"negative_scores = {negative_scores[count - 1]}")
        # Calculate EER from the scores.
        eer = self.calculate_EER(positive_scores, negative_scores)
        print(f"Equal Error Rate (EER): {eer * 100:.2f}%")
        return eer * 100
