import torchaudio
import os

aviv_raw_path = "raw/aviv"
rachel_raw_path = "raw/rachel"
aviv_resampled_path = "resampled/aviv"
rachel_resampled_path = "resampled/rachel"


def resample_audio_files(input_dir, output_dir, target_sr=16000):
    """Resamples all WAV files in input_dir to target_sr and saves them to output_dir."""
    os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist

    for filename in os.listdir(input_dir):
        if filename.endswith(".wav"):  # Process only WAV files
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # Load audio
            waveform, sample_rate = torchaudio.load(input_path)

            # Resample if needed
            if sample_rate != target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
                waveform = resampler(waveform)

            # Save the resampled audio
            torchaudio.save(output_path, waveform, target_sr)
            print(f"Resampled: {input_path} -> {output_path}")


def main():
    resample_audio_files(aviv_raw_path, aviv_resampled_path)
    resample_audio_files(rachel_raw_path, rachel_resampled_path)


if __name__ == '__main__':
    main()
