# Voice Conversion With Just Nearest Neighbors
## Links:
- Arvix Paper: https://arxiv.org/abs/2305.18975
- Official Code Repo: https://github.com/bshall/knn-vc

## Overview
In the following repo we are going to implement and analyze the kNN-VC method, introduced in Voice Conversion with Just Nearest Neighbors in 2023.
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

| Checkpoint      | WER (%)    | CER(%)     | EER(%)     |
|:----------|:---------:|:----------:|:-----------:|
| original kNN-VC with prematched HiFiGAN     | Banana    | Cherry    | Orange    |
| our kNN-VC with prematched HiFiGAN        | Cat       | Elephant  | Mouse     |
| original kNN-VC with regular HiFiGAN       | Blue      | Green     | Yellow    |
| our kNN-VC with regular HiFiGAN       | Two       | Three     | Four      |

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
- `--device`: Inference device: `"cuda"` (default) or `"cpu"` . If you don’t have a GPU, make sure to pass `--device cpu`.  

- `--k`: Number of nearest neighbors (default: `4`)  

- `--prematched`: Use the vocoder trained on **prematched** features (recommended). Pass `--prematched` to enable it  

- `--use_custom_path`: Use your **locally trained vocoder** instead of the default. Pass `--use_custom_path` if you want to use our custom HiFi-GAN checkpoint  

To run the code:
<pre>python run_pipline.py \
  --src path/to/source.wav \
  --refs path/to/target1.wav path/to/target2.wav path/to/target3.wav \
  --out outputs/converted.wav \
  --device cpu \
  --prematched \
  --use_custom_path</pre>

## Training
We follow the typical encoder-converter-vocoder setup for voice conversion. The encoder is WavLM, the converter is k-nearest neighbors regression, and vocoder is HiFiGAN. The only component that requires training is the vocoder.

### HiFiGAN training:
For training we require the same dependencies as the original HiFiGAN training. But no worries if you install the provided requirements you should be ok :)

#### Step 0 - Download the Dataset:
For training we use the LibraSpeech dataset that includes approximately 1000 hours of 16kHz read English speech. 
In particular, we used the train-clean-100 and the dev-clean dataset.
Here, the link to download them: https://www.openslr.org/12

##### Step 1 - Pre-Processing
Before training HiFi-GAN, you must precompute WavLM features for your dataset using prematch_dataset.py. This step creates .pt feature files for each .flac utterance and optionally performs prematching to improve the final synthesis quality.
As explained in the paper, prematching reduces vocoder artifacts and increases intelligibility by generating target features conditioned on pre-aligned top-k reference frames. You can toggle prematching behavior using the --prematch flag.

To precompute WavLM features run:
<pre>
  python prematch_dataset.py --librispeech_path dataset --out_path dataset/prematched_features --topk 4 --matching_layer 6 --synthesis_layer 6 --prematch --device cpu --resume
</pre>

##### Available Arguments:
| Flag                 | Description                                                                                  |
|----------------------|----------------------------------------------------------------------------------------------|
| `--librispeech_path` | Path to the root folder of your dataset (LibriSpeech-style `.flac` subfolders). Inside this folder, the code will look for test-clean-100 and dev-clean folder.|
| `--out_path`         | Where to save the `.pt` feature files (mirrors LibriSpeech structure).                      |
| `--topk`             | Number of nearest neighbors (frames) to retrieve for prematching.                           |
| `--matching_layer`   | Which WavLM layer to use for similarity computation.                                         |
| `--synthesis_layer`  | Which WavLM layer to save as the output features.                                            |
| `--device`           | Device to run on (`cuda` or `cpu`).                                                          |
| `--prematch`         | Enable prematching logic (optional). If omitted, standard non-prematched features are used. |
| `--resume`           | Skip already processed files to allow resuming from interrupted runs.                       |
| `--seed`             | (Optional) Set random seed for reproducibility (default: 123).                              |

#### Step 2: Train HiFi-GAN Vocoder
Once you've precomputed the WavLM features (with or without --prematch), you're ready to train the HiFi-GAN vocoder.
We use the original hifigan/train.py script provided by the authors. You can launch training with:

<pre>
  python -m hifigan.train \
  --audio_root_path dataset/ \
  --feature_root_path prematched_features/ \
  --input_training_file data_splits/wavlm-hifigan-train.csv \
  --input_validation_file data_splits/wavlm-hifigan-valid.csv \
  --checkpoint_path outputs/ \
  --fp16 True \
  --config hifigan/config_v1_wavlm.json \
  --stdout_interval 25 \
  --training_epochs 2 \
  --fine_tuning
</pre>

Note: In the original paper, they trained up to 2.5M updates. We didn’t reach that due to limited compute — we hope you can! :)
##### Available Arguments:

| Flag                      | Description                                                                                         |
|---------------------------|-----------------------------------------------------------------------------------------------------|
| `--group_name`            | (Optional) Name of experiment group. Useful for logging and organizing multiple runs.              |
| `--audio_root_path`       | **Required.** Path to the original raw audio files (`.flac` or `.wav`).                            |
| `--feature_root_path`     | **Required.** Path to the corresponding precomputed WavLM `.pt` features.                          |
| `--input_training_file`   | Path to CSV file listing training utterances. Default: `LJSpeech-1.1/training.txt`.                |
| `--input_validation_file` | Path to CSV file listing validation utterances. Default: `LJSpeech-1.1/validation.txt`.            |
| `--checkpoint_path`       | Directory to store model checkpoints. Default: `cp_hifigan`.                                       |
| `--config`                | Path to the HiFi-GAN configuration file (e.g., `hifigan/config_v1_wavlm.json`).                    |
| `--training_epochs`       | Total number of training epochs. Default: `1500`.                                                   |
| `--stdout_interval`       | Print logs to console every N steps. Default: `5`.                                                  |
| `--checkpoint_interval`   | Save model checkpoints every N steps. Default: `5000`.                                              |
| `--summary_interval`      | Write tensorboard summary every N steps. Default: `25`.                                             |
| `--validation_interval`   | Run validation loop every N steps. Default: `1000`.                                                 |
| `--fp16`                  | Enable automatic mixed precision training (faster & memory efficient). Default: `False`.           |
| `--fine_tuning`           | Load weights from previous checkpoint and resume training. Use for transfer learning setups.        |

#### Monotoring Results:
During training, model checkpoints and logs are saved to the specified output directory. You can monitor the training process in real-time using **TensorBoard**, including tracking generator loss, mel-spectrogram loss, and discriminator performance.
To launch TensorBoard, run the following command and replace `path_to_logs` with your actual checkpoint log directory:
<pre> tensorboard --logdir path_output_logs </pre>

## Evaluation
We 


