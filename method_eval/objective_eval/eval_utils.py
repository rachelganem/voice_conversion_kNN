import os
import random


def get_source_and_target_speaker_from_filename(filename):
    """
    Extract the source and target speaker IDs from the filename.

    Parameters:
    - filename: str, the filename of the audio file (e.g., "61-70968-0029_to_121.wav")

    Returns:
    - source: str, the source speaker ID extracted from the filename
    - target: str, the target speaker ID extracted from the filename
    """
    source = filename.split("-")[0]
    target = filename.split("_to_")[-1].split(".")[0]
    return source, target


def get_target_ref_file(target_path, use_first=False):
    """
    Retrieve a random .flac file for the target speaker.

    Parameters:
    - target_path: str, the path to the target speaker folder
    - use_first: bool, whether to return the first .flac file found (defaults to False)

    Returns:
    - str, path to a random .flac file (absolute or relative depending on configuration)
    """
    flac_files = []
    for root, dirs, files in os.walk(target_path):
        for file in files:
            if file.endswith(".flac"):
                flac_files.append(os.path.join(root, file))
                # If use_first is True, return immediately
                if use_first:
                    return flac_files[0]

    if flac_files:
        return random.choice(flac_files)
    else:
        raise FileNotFoundError(f"No .flac files found in {target_path}")


def create_speaker_dict(data_dir):
    """
    Create a dictionary with speaker ID as key and a random .flac file as value.

    Parameters:
    - data_dir: str, the directory containing speaker folders

    Returns:
    - dict: speaker_id -> random .flac file path
    """
    speaker_flac_dict = {}
    for speaker_folder in os.listdir(data_dir):
        speaker_path = os.path.join(data_dir, speaker_folder)
        if os.path.isdir(speaker_path):
            try:
                random_flac_file = get_target_ref_file(speaker_path)
                speaker_flac_dict[speaker_folder] = random_flac_file
            except FileNotFoundError as e:
                print(f"Warning: {e}")

    return speaker_flac_dict


def get_random_negative_sample(speaker_dict, source_id, target_id):
    """
    Choose a random item from speaker_dict where the key is different from both the source_id and target_id.

    Parameters:
    - speaker_dict: dict, mapping speaker_id to .flac file path
    - source_id: str, the source speaker ID to exclude
    - target_id: str, the target speaker ID to exclude

    Returns:
    - str, path to a random negative .flac file
    """
    # Filter the dictionary to exclude entries where the key is either source_id or target_id
    filtered_speaker_dict = {key: value for key, value in speaker_dict.items() if key != source_id and key != target_id}

    if not filtered_speaker_dict:
        raise ValueError("No valid speakers left for negative sampling (both source and target speakers are excluded).")

    # Choose a random item from the filtered dictionary
    random_key = random.choice(list(filtered_speaker_dict.keys()))
    return filtered_speaker_dict[random_key]
