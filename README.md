# Voice Conversion With Just Nearest Neighbors
Links:
- Arvix Paper: https://arxiv.org/abs/2305.18975
- Official Code Repo: https://github.com/bshall/knn-vc

We want to be able to convert source voice to target voice without additional training.
The method includes three steps:
1. Encode source and reference utterances uning WavLM
2. Each source feature is assigned to the mean of the k closest features from the reference.
3. The resulting feature sequence is then vocoded with HiFi-GAN to arrive at the converted waveform output.

In the following code repo we will implement by ourselves the paper above, but we'll try to training the encoder and the decoder with a different dataset.


Requirements:
torch v2.0 or greater
torchaudio
numpy
python v3.10 or greater


python -m hifigan.train --audio_root_path dataset/ --feature_root_path prematched_features/ --input_training_file data_splits/wavlm-hifigan-train.csv --input_validation_file data_splits/wavlm-hifigan-valid.csv --checkpoint_path outputs/ --fp16 True --config hifigan/config_v1_wavlm.json --stdout_interval 25 --training_epochs 2 --fine_tuning

TRAINING COMMANDS

sh rachelganem@c-002.cs.tau.ac.il
C!ccia01
cd /home/yandex/APDL2425a/group_13/try/knn-vc-ref/
bash
conda activate audio
export TORCH_HOME=/vol/scratch/rachelganem/torch_cache
sbatch job.slurm
//check if job is running
squeue -u rachelganem


Requirements:
python 3.10.6
pip install torch==2.1.2+cu121 torchaudio==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas fastprogress matplotlib librosa soundfile scipy tensorboard
pip install numpy==1.26.4
pip install git+https://github.com/openai/whisper.git
pip install jiwer
pip install speechbrain
pip install scikit-learn

#!/bin/bash
#SBATCH --job-name=eval_dataset                # Name of your job
#SBATCH --output=logs/eval_dataset_%j.out       # Output log file (%j is replaced by the job ID)
#SBATCH --error=logs/eval_dataset_%j.err        # Error log file
#SBATCH --partition=studentkillable            # Partition with GPU nodes (Titan XP)
#SBATCH --gres=gpu:1                           # Request 3 GPUs
#SBATCH --nodes=1                              # Use one node
#SBATCH --ntasks=1                             # One task (process)
#SBATCH --cpus-per-task=4                      # Number of CPU cores per task
#SBATCH --time=24:00:00                        # Maximum runtime (hh:mm:ss)

# Load required modules
# module load anaconda3

# Activate your conda environmen
# source activate audio
# Create logs directory
# mkdir -p logs

# Set TORCH_HOME if needed
export TORCH_HOME=/vol/scratch/rachelganem/torch_cache
export HOME=/tmp
export XDG_CACHE_HOME=/vol/scratch/rachelganem/tmp_cache

# Specify which GPUs you want visible (here GPUs 1, 2, and 3)
#export CUDA_VISIBLE_DEVICES=1,2,3

# Run your training script with the necessary configuration files
python -m method_eval.build_eval_set 