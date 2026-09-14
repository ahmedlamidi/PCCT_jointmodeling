#!/bin/bash -l
#SBATCH --job-name=pcct-wgan
#SBATCH -p Quick
#SBATCH --cpus-per-task=4
#SBATCH --time=22:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ahmedlamidi@usf.edu
#SBATCH --output=logs/wgan_%j.out
#SBATCH --error=logs/wgan_%j.err

# Morovati et al. 2025 baseline (residual WGAN-GP). Single device, as above.
# 40 epochs may not fit one 22 h allocation. --resume makes that safe: if the job
# hits the time limit, resubmit THIS SAME SCRIPT and it continues from the last
# checkpoint (saved every 10k iterations, atomically). Evaluation runs only once
# training reaches the end.
# lam_perc defaults to 0: their perceptual term needs a ViT trained on PCCT data
# (they explicitly do NOT use ImageNet pretraining), so there is no checkpoint to
# load. With random weights that term is noise. Train a ViT and pass --vit_ckpt
# if you want it, and say which you did in the write-up.

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source /apps/anaconda3/etc/profile.d/conda.sh
CONDA_ENV=proj
conda activate "$CONDA_ENV"
pip install -r requirements.txt

export ARM=${ARM:-baseline3d_pu_matched}
EPOCHS=${EPOCHS:-40}    # Morovati et al.: 40 epochs. ~1.09M iterations on our data at batch 64
BATCH=${BATCH:-64}
LAM_PERC=${LAM_PERC:-0}

cd src/baseline_wgan

echo "=== train WGAN-GP baseline on $ARM ==="
srun --ntasks=1 python -u train.py \
    --arm "$ARM" --target Y \
    --epochs "$EPOCHS" --batch "$BATCH" --resume \
    --lam_perc "$LAM_PERC"

echo "=== evaluate on the held-out TEST split of $ARM ==="
srun --ntasks=1 python -u evaluate.py \
    --ckpt "../../outputs/wgan_${ARM}_Y/ckpt.pt" \
    --arms "$ARM"
