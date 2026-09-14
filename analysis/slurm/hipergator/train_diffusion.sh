#!/bin/bash
#SBATCH --job-name=pcct-edm
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --time=72:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ahmedlamidi@usf.edu
#SBATCH --output=logs/edm_%j.out
#SBATCH --error=logs/edm_%j.err
#
# HiPerGator EDM diffusion. hpg-turin = L4 (24 GB) -- ample for a ~15M-param UNet
# on 16x16 patches, and the least contended GPU. For a B200 instead:
#   sbatch --partition=hpg-b200 ...   (the cu128 torch build supports both)
# 72 h fits burst QOS (96 h cap) and the 14-day GPU partition cap. If it still
# times out, resubmit: --resume continues from the last checkpoint.

cd "$SLURM_SUBMIT_DIR"
GROUP=${GROUP:-$(id -gn)}
ENV=${PCCT_ENV:-/blue/$GROUP/$USER/conda/envs/pcct}
module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate "$ENV"
nvidia-smi -L || true

export ARM=${ARM:-baseline3d_pu_matched}   # exported: smoke_test.py reads it
ITERS=${ITERS:-200000}
BATCH=${BATCH:-128}
cd src/diffusion

echo "=== smoke test: shape bugs surface here in ~1 min ==="
srun python -u smoke_test.py || { echo "SMOKE TEST FAILED - stopping"; exit 1; }

echo "=== train EDM on $ARM  $(date) ==="
srun python -u train.py --arm "$ARM" --target Y \
    --iters "$ITERS" --batch "$BATCH" --resume --preload

echo "=== evaluate on the held-out TEST split  $(date) ==="
srun python -u coverage.py --ckpt "../../outputs/edm_${ARM}_Y/ckpt.pt" \
    --arms "$ARM" --nsamp 256
