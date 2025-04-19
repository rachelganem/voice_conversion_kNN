# Voice Conversion With Just Nearest Neighbors

## Links:

- Our github repo: https://github.com/rachelganem/voice_conversion_kNN
- Arvix Paper: https://arxiv.org/abs/2305.18975
- Official Code Repo: https://github.com/bshall/knn-vc
- LibriSpeech dataset: https://www.openslr.org/12
- Our Medium Post:

## Overview

In the following repo we are going to implement and analyze the kNN-VC method, introduced in Voice Conversion with Just
Nearest Neighbors in 2023.
kNN-VC is a non-parametic approach to any-to-any voice conversion that includes three steps:

1. Encoding raw audio of source and target spakers to self-supervised speech representation using the WavLM model
2. We repleace each frame of the source with the mean of the k closest features from the reference using kNN algorithm.
3. We convert the resulting feature sequence to waveform (.wav) using Hi-FiGan as vocoder.

## Checkpoints

We provide five checkpoints:

1. The frozen WavLM Encoder from the original WavLM authors
2. The HiFiGAN vocoder trained on layer 6 of WavLM features of the original repo - They did 2.5M steps!
3. The HiFiGAN vocoder trained on layer 6 of WavLM features that we trained
4. The HiFiGAN vocoder trained on prematched layer 6 of WaveLM features of the original repo (The best!)
5. The HiFiGAN vocoder trained on prematched layer 6 of WaveLM features that we trained

We evaluate our checkpoints on LibriSpeech test-clean dataset. Here, is the performance summarized:

| Checkpoint                              | WER (%) | CER(%) | EER(%) |
|:----------------------------------------|:-------:|:------:|:------:|
| original kNN-VC with prematched HiFiGAN |  27.16  |  9.78  |  9.12  |
| our kNN-VC with prematched HiFiGAN      |  28.29  | 10.45  |  8.69  |
| original kNN-VC with regular HiFiGAN    |  28.05  | 10.24  |  6.35  |
| our kNN-VC with regular HiFiGAN         |  28.23  | 10.38  |  8.04  |

## Requirements

This project requires Python 3.10.
To install all dependencies (for both inference and training), run:
<pre> pip install -r requirements.txt</pre>

## Inference

To run the voice conversion pipeline, use the script `run_pipline.py` with the appropriate parameters.

### Required Arguments:

- `--src`: Path to the source audio file (`.wav`)
- `--refs`: List of target/reference audio files (`.wav`) . We recommend providing at least 5 target utterances.
- `--out`: Output path for the converted `.wav` file

### Optional Arguments:

- `--device`: Inference device: `"cuda"` (default) or `"cpu"` . If you don’t have a GPU, make sure to pass
  `--device cpu`.

- `--k`: Number of nearest neighbors (default: `4`)

- `--prematched`: Use the vocoder trained on **prematched** features (recommended). Pass `--prematched` to enable it

- `--use_custom_path`: Use your **locally trained vocoder** instead of the default. Pass `--use_custom_path` if you want
  to use our custom HiFi-GAN checkpoint

To run the code:
<pre>python run_pipline.py --src path/to/source.wav -refs path/to/target1.wav path/to/target2.wav path/to/target3.wav --out outputs/converted.wav --device cpu --prematched --use_custom_path</pre>

## Training

We follow the typical encoder-converter-vocoder setup for voice conversion. The encoder is WavLM, the converter is
k-nearest neighbors regression, and vocoder is HiFiGAN. The only component that requires training is the vocoder.

### HiFiGAN training:

For training we require the same dependencies as the original HiFiGAN training. But no worries if you install the
provided requirements you should be ok :)

#### Step 0 - Download the Dataset:

For training we use the LibraSpeech dataset that includes approximately 1000 hours of 16kHz read English speech.
In particular, we used the train-clean-100 and the dev-clean dataset.
Here, the link to download them: https://www.openslr.org/12

#### Step 1 - Pre-Processing

Before training HiFi-GAN, you must precompute WavLM features for your dataset using prematch_dataset.py. This step
creates .pt feature files for each .flac utterance and optionally performs prematching to improve the final synthesis
quality.
As explained in the paper, prematching reduces vocoder artifacts and increases intelligibility by generating target
features conditioned on pre-aligned top-k reference frames. You can toggle prematching behavior using the --prematch
flag.

To precompute WavLM features run:
<pre>
  python prematch_dataset.py --librispeech_path dataset --out_path dataset/prematched_features --topk 4 --matching_layer 6 --synthesis_layer 6 --prematch --device cpu --resume
</pre>

##### Available Arguments:

| Flag                 | Description                                                                                                                                                     |
|----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--librispeech_path` | Path to the root folder of your dataset (LibriSpeech-style `.flac` subfolders). Inside this folder, the code will look for test-clean-100 and dev-clean folder. |
| `--out_path`         | Where to save the `.pt` feature files (mirrors LibriSpeech structure).                                                                                          |
| `--topk`             | Number of nearest neighbors (frames) to retrieve for prematching.                                                                                               |
| `--matching_layer`   | Which WavLM layer to use for similarity computation.                                                                                                            |
| `--synthesis_layer`  | Which WavLM layer to save as the output features.                                                                                                               |
| `--device`           | Device to run on (`cuda` or `cpu`).                                                                                                                             |
| `--prematch`         | Enable prematching logic (optional). If omitted, standard non-prematched features are used.                                                                     |
| `--resume`           | Skip already processed files to allow resuming from interrupted runs.                                                                                           |
| `--seed`             | (Optional) Set random seed for reproducibility (default: 123).                                                                                                  |

#### Step 2: Train HiFi-GAN Vocoder

Once you've precomputed the WavLM features (with or without --prematch), you're ready to train the HiFi-GAN vocoder.
We use the original hifigan/train.py script provided by the authors. You can launch training with:

<pre>
  python -m hifigan.train --audio_root_path dataset/ --feature_root_path prematched_features/ --input_training_file data_splits/wavlm-hifigan-train.csv --input_validation_file data_splits/wavlm-hifigan-valid.csv --checkpoint_path outputs/ --fp16 True --config hifigan/config_v1_wavlm.json --stdout_interval 25 --training_epochs 2 --fine_tuning
</pre>

Note: In the original paper, they trained up to 2.5M updates. We didn’t reach that due to limited compute — we hope you
can! :)

##### Available Arguments:

| Flag                      | Description                                                                                  |
|---------------------------|----------------------------------------------------------------------------------------------|
| `--group_name`            | (Optional) Name of experiment group. Useful for logging and organizing multiple runs.        |
| `--audio_root_path`       | **Required.** Path to the original raw audio files (`.flac` or `.wav`).                      |
| `--feature_root_path`     | **Required.** Path to the corresponding precomputed WavLM `.pt` features.                    |
| `--input_training_file`   | Path to CSV file listing training utterances. Default: `LJSpeech-1.1/training.txt`.          |
| `--input_validation_file` | Path to CSV file listing validation utterances. Default: `LJSpeech-1.1/validation.txt`.      |
| `--checkpoint_path`       | Directory to store model checkpoints. Default: `cp_hifigan`.                                 |
| `--config`                | Path to the HiFi-GAN configuration file (e.g., `hifigan/config_v1_wavlm.json`).              |
| `--training_epochs`       | Total number of training epochs. Default: `1500`.                                            |
| `--stdout_interval`       | Print logs to console every N steps. Default: `5`.                                           |
| `--checkpoint_interval`   | Save model checkpoints every N steps. Default: `5000`.                                       |
| `--summary_interval`      | Write tensorboard summary every N steps. Default: `25`.                                      |
| `--validation_interval`   | Run validation loop every N steps. Default: `1000`.                                          |
| `--fp16`                  | Enable automatic mixed precision training (faster & memory efficient). Default: `False`.     |
| `--fine_tuning`           | Load weights from previous checkpoint and resume training. Use for transfer learning setups. |

#### Monotoring Results:

During training, model checkpoints and logs are saved to the specified output directory. You can monitor the training
process in real-time using **TensorBoard**, including tracking generator loss, mel-spectrogram loss, and discriminator
performance.
To launch TensorBoard, run the following command and replace `path_to_logs` with your actual checkpoint log directory:
<pre> tensorboard --logdir path_output_logs </pre>

## Evaluation

We evaluate our model with both objective and objective metrics.
We implemented a module called method_eval that contains the script to build the test dataset, to perform the objective
and subjective evaluation.

### Step 1 - Download Test Dataset:

Download the test-clean dataset from LibraSpeech. It can be find here: https://www.openslr.org/12

### Step 2 - Build Evaluation Set:

To build the evaluation set, run the `build_eval_set.py` script located in the `method_eval/` folder. This script
performs the following:

1. Sampling: Randomly selects a specified number of speakers (default: 40), and a fixed number of utterances per
   speaker (default: 5).
2. CSV Generation: Creates eval_set.csv, listing all selected speaker IDs and their corresponding .flac utterance paths.
3. Transcript Extraction: Searches for corresponding transcript files and writes them into converted/transcripts.csv.
   This is required for intelligibility evaluation.
4. Inference: Converts each source utterance to all other target speakers in four settings (see “Checkpoints”):
   Prematched vocoder (pretrained & custom) and Regular vocoder (pretrained & custom). This behavior can be modified
   using flags like --run_all, --use_custom_path, and --prematched.

Note: if `--run_all` flag is set to true, we ignore the `--use_custom_path` and `--prematched` flags

To run it:
<pre>python method_eval/build_eval_set.py --test_path ../data/dataset/raw/test-clean --csv_path eval_set.csv --num_speakers 40 --num_utterances 5 --seed 42 --device cuda --run_all </pre>

##### Available Arguments:

| Flag                | Description                                                            |
|---------------------|------------------------------------------------------------------------|
| `--test_path`       | Path to the root folder of `test-clean` (LibriSpeech-style structure). |
| `--csv_path`        | Output path to save `eval_set.csv`.                                    |
| `--device`          | Device for inference (`cuda` or `cpu`).                                |
| `--run_all`         | Run all 4 configurations (custom/prematched × original/normal).        |
| `--use_custom_path` | Use your own trained vocoder weights instead of pretrained.            |
| `--prematched`      | Run only on prematched or normal mode (default: True).                 |
| `--num_speakers`    | Number of speakers to sample from the test set (default: 40).          |
| `--num_utterances`  | Number of utterances per sampled speaker (default: 5).                 |
| `--seed`            | Random seed for reproducibility (optional, default: 123).              |
| `--k`               | Number of nearest neighbors to use in kNN-VC (default: 4).             |
| `--output_dir`      | Base output directory                                                  |

### Step 3 - Run Objective Evaluation:

To evaluate objective metrics such as intelligibility (WER/CER) and/or speaker similarity, run the `eval_objective.py`
script located in `method_eval\objective_eval` folder. The --test_path argument is always required, while the
--transcript_csv is required only for intelligibility evaluation

You can choose which evaluation(s) to run using the --eval flag. You may run only one, or both at the same time.
Run intelligibility only:
<pre>
python -m method_eval.objective_eval.eval_objective --eval intelligibility --transcript_csv method_eval/converted/transcripts.csv --converted_path method_eval/converted/prematched --output_path results/objective_metrics.csv --test_path data/dataset/raw/test-clean
</pre>

Run speaker similarity only:
<pre> python -m method_eval.objective_eval.eval_objective --eval speaker_similarity --converted_path method_eval/converted/prematched --output_path results/objective_metrics.csv --test_path data/dataset/raw/test-clean
</pre>

Run both evaluations:
<pre>
python -m method_eval.objective_eval.eval_objective --eval intelligibility speaker_similarity --transcript_csv method_eval/converted/transcripts.csv --converted_path method_eval/converted/prematched  --output_path results/objective_metrics.csv --test_path data/dataset/raw/test-clean --device cuda
</pre>

##### Available Arguments:

| Flag               | Description                                                                                                 |
|--------------------|-------------------------------------------------------------------------------------------------------------|
| `--test_path`      | (Required) Path to the root folder of `test-clean` (LibriSpeech-style structure).                           |
| `--eval`           | (Required)Evaluation types to run. Options: intelligibility, speaker_similarity, or both (space-separated). |
| `--converted_path` | (Required) Path to the folder containing converted .wav files.                                              |
| `--output_path`    | (Required) Path to the output file where results will be appended.                                          |
| `--transcript_csv` | Path to the transcript CSV file (required for intelligibility evaluation).                                  |
| `--whisper_model`  | Whisper model size to use. Options: tiny, base, small, medium, large (default: base).                       |
| `--device`         | Device to run models on. Default: cpu                                                                       |

### Step 4 - Subjective Evaluation:

We created a Google Form to conduct our subjective evaluation, from which we calculated both Mean Opinion Score (MOS)
and Speaker Similarity (SIM) based on participant responses.
If you'd like to participate in our survey, you can do so using the following link: https://forms.gle/q85vmwZBCC6i4Vfs6

# Repository Structure

<pre>
project-root/
├── README.md
├── requirements.txt
├── run_pipeline.py

├── data/
│   ├── data_splits/                         # Train/validation splits
│   │   ├── wavlm-hifigan-train.csv
│   │   └── wavlm-hifigan-valid.csv
│   └── dataset/                             
│   │   ├── prematched
│   │   |   └── normal
│   │   |   └── prematched
│   │   └── raw                              # Raw/test-clean LibriSpeech-style audio
│   │   |   └── test-clean
│   │   |   └── train-clean-100
│   │   |   └── dev-clean
|
├── method_eval/
|   └── build_eval_set.py
|   └── eval_set.csv                         # evaluation set - build when running build_eval_set.py
|
│   ├── converted/                           # Output of knnVC method on test-clean
│   │   ├── normal/
│   │   ├── prematched/
│   │   └── transcripts.csv                  # For intelligibility eval
│
│   ├── objective_eval/
│   │   ├── resemblyzer/                     # Adapted resemblyzer module for extracting x-vector from audio
│   │   │   ├── audio.py
│   │   │   ├── hparams.py
│   │   │   ├── pretrained.pt
│   │   │   └── voice_encoder.py
│   │   ├── eval_objective.py
│   │   ├── eval_utils.py
│   │   ├── evaluate_intelligibility.py
│   │   └── evaluate_speaker_similarity.py
│ 
|
├── outputs/
│   └── logs/                               #training logs
│       ├── normal/
│       ├── prematch_training1/
│       └── prematch_training2/

├── src/
│   ├── hifigan/
│   │   ├── config_v1_wavlm.json
│   │   ├── meldataset.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── utils.py
│
│   ├── knnvc/
│   │   ├── hubconf.py
│   │   ├── knnvc_utils.py
│   │   ├── matcher.py
│   │   └── prematch_dataset.py
│
│   ├── wavlm/
│   │   ├── modules.py
│   │   └── WavLM.py
│
│   └── weights/                         # training weights
│       ├── weights_config.json
│       ├── normal/
│       │   ├── g_00068000.pt
│       │   └── g_00118000.pt
│       └── prematched/
│           ├── g_00062000.pt
│           └── g_00121000.pt

</pre>

# Acknowledgments

Parts of code for this project are adapted from the following repositories

- HiFiGAN: https://github.com/jik876/hifi-gan
- WavLM: https://github.com/microsoft/unilm/tree/master/wavlm
- kNN-VC official repo: https://github.com/bshall/knn-vc/tree/master
- resemblyzer: https://github.com/resemble-ai/Resemblyzer

# Citation

<pre>
@inproceedings{baas2023knnvc,
author={Matthew Baas and Benjamin van Niekerk and Herman Kamper},
title={Voice Conversion With Just Nearest Neighbors},
year=2023,
booktitle={Interspeech},
}

</pre>
