from scipy.ndimage import binary_dilation
from .hparams import *
from pathlib import Path
from typing import Optional, Union
import numpy as np
import librosa
import struct

int16_max = (2 ** 15) - 1


def preprocess_wav(fpath_or_wav: Union[str, Path, np.ndarray], source_sr: Optional[int] = None):
    if isinstance(fpath_or_wav, (str, Path)):
        wav, source_sr = librosa.load(str(fpath_or_wav), sr=None)
    else:
        wav = fpath_or_wav

    if source_sr is not None:
        wav = librosa.resample(wav, orig_sr=source_sr, target_sr=sampling_rate)

    wav = normalize_volume(wav, audio_norm_target_dBFS, increase_only=True)
    wav = trim_long_silences(wav)

    return wav


def wav_to_mel_spectrogram(wav):
    frames = librosa.feature.melspectrogram(
        y=wav,
        sr=sampling_rate,
        n_fft=int(sampling_rate * mel_window_length / 1000),
        hop_length=int(sampling_rate * mel_window_step / 1000),
        n_mels=mel_n_channels
    )
    return frames.astype(np.float32).T


def trim_long_silences(wav):
    """
    Replaces WebRTC VAD with an energy-based silence detector using librosa.effects.split.
    It trims long silent sections while respecting your VAD config.
    """
    frame_length = int(sampling_rate * vad_window_length / 1000)
    hop_length = frame_length // 2

    # librosa.split returns non-silent intervals based on decibel threshold
    intervals = librosa.effects.split(wav, top_db=40, frame_length=frame_length, hop_length=hop_length)

    if len(intervals) == 0:
        return wav  # fallback: don't trim anything

    # Reconstruct signal from voiced intervals
    voiced_wav = np.concatenate([wav[start:end] for start, end in intervals])

    # Pad or smooth like the original logic (binary dilation)
    voiced_mask = np.zeros(len(wav), dtype=bool)
    for start, end in intervals:
        voiced_mask[start:end] = True

    # Smooth the voiced mask to avoid hard cuts
    smoothing_width = frame_length * vad_max_silence_length
    if smoothing_width > 1:
        kernel = np.ones(smoothing_width)
        voiced_mask = binary_dilation(voiced_mask, kernel)

    return wav[voiced_mask]


def normalize_volume(wav, target_dBFS, increase_only=False, decrease_only=False):
    if increase_only and decrease_only:
        raise ValueError("Both increase_only and decrease_only are set")
    rms = np.sqrt(np.mean((wav * int16_max) ** 2))
    wave_dBFS = 20 * np.log10(rms / int16_max)
    dBFS_change = target_dBFS - wave_dBFS
    if dBFS_change < 0 and increase_only or dBFS_change > 0 and decrease_only:
        return wav
    return wav * (10 ** (dBFS_change / 20))
