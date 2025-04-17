import os
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cosine
from sklearn.metrics import roc_curve
from tqdm import tqdm

import eval_utils as ut
from resemblyzer import voice_encoder as ve
from resemblyzer import audio as a

class SpeakerSimilarityEvaluator:
    def __init__(self, test_data_path, device):
        self.data_path = test_data_path
        self.encoder = ve.VoiceEncoder(device)

    def extract_x_vector(self, audio_path):
        wav = a.preprocess_wav(Path(audio_path))
        return self.encoder.embed_utterance(wav)

    def cosine_similarity(self, vector1, vector2):
        return 1 - cosine(vector1, vector2)

    def calculate_EER(self, positive_similarities, negative_similarities):
        similarities = positive_similarities + negative_similarities
        labels = [1] * len(positive_similarities) + [0] * len(negative_similarities)
        fpr, tpr, thresholds = roc_curve(labels, similarities)
        eer_index = np.nanargmin(np.abs(fpr - (1 - tpr)))
        return fpr[eer_index]

    def evaluate(self, converted_path):
        speaker_dict = ut.create_speaker_dict(self.data_path)
        positive_scores = []
        negative_scores = []
        count = 0

        for root, dirs, files in os.walk(converted_path):
            for file in tqdm(files, desc="Evaluating files", unit="file"):
                if file.endswith(".wav"):
                    count += 1
                    converted_audio_path = os.path.join(root, file)
                    source_id, target_id = ut.get_source_and_target_speaker_from_filename(file)

                    converted_target_vector = self.extract_x_vector(converted_audio_path)
                    positive_target_vect = self.extract_x_vector(speaker_dict[target_id])
                    negative_sample_path = ut.get_random_negative_sample(speaker_dict, source_id, target_id)
                    negative_target_vect = self.extract_x_vector(negative_sample_path)

                    positive_scores.append(self.cosine_similarity(positive_target_vect, converted_target_vector))
                    negative_scores.append(self.cosine_similarity(negative_target_vect, converted_target_vector))
                    print(f"count = {count}")
                    print(f"positive_scores = {positive_scores[-1]}")
                    print(f"negative_scores = {negative_scores[-1]}")

        eer = self.calculate_EER(positive_scores, negative_scores)
        print(f"Equal Error Rate (EER): {eer * 100:.2f}%")
        return eer * 100
