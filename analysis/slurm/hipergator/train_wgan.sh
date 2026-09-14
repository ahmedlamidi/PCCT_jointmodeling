#!/bin/bash
#SBATCH --job-name=pcct-wgan
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --time=72:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ahmedlamidi@usf.edu
#SBATCH --output=logs/wgan_%j.out
#SBATCH --error=logs/wgan_%j.err
#
# HiPerGator Morovati et al. 2025 baseline (residual WGAN-GP). 40 epochs to match
# their schedule = ~1.09M iterations on this data; may exceed one 72 h job. If it
# times out, RESUBMIT THIS SCRIPT -- --resume continues from the last checkpoint,
# and evaluation runs only once training reaches the final iteration.

cd "$SLURM_SUBMIT_DIR"
GROUP=${GROUP:-$(id -gn)}
ENV=${PCCT_ENV:-/blue/$GROUP/$USER/conda/envs/pcct}
module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate "$ENV"
nvidia-smi -L || true

export ARM=${ARM:-baseline3d_pu_matched}
EPOCHS=${EPOCHS:-40}
BATCH=${BATCH:-64}
LAM_PERC=${LAM_PERC:-0}   # their perceptual ViT is trained from scratch; none exists
cd src/baseline_wgan

echo "=== train WGAN-GP on $ARM  $(date) ==="
srun python -u train.py --arm "$ARM" --target Y \
    --epochs "$EPOCHS" --batch "$BATCH" --resume --lam_perc "$LAM_PERC"

echo "=== evaluate on the held-out TEST split  $(date) ==="
srun python -u evaluate.py --ckpt "../../outputs/wgan_${ARM}_Y/ckpt.pt" --arms "$ARM"
