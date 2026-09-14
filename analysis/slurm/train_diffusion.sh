#!/bin/bash -l
#SBATCH --job-name=pcct-edm
#SBATCH -p Quick
#SBATCH --cpus-per-task=4
#SBATCH --time=22:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ahmedlamidi@usf.edu
#SBATCH --output=logs/edm_%j.out
#SBATCH --error=logs/edm_%j.err

# gpu:1, not gpu:2 -- train.py is single-device (no DDP / DataParallel), so a
# second GPU would sit idle and only lengthen the queue wait. Add DDP first if
# you want 2.

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source /apps/anaconda3/etc/profile.d/conda.sh
CONDA_ENV=proj
conda activate "$CONDA_ENV"
pip install -r requirements.txt

export ARM=${ARM:-baseline3d_pu_matched}   # exported: smoke_test.py reads it from the environment
ITERS=${ITERS:-200000}
BATCH=${BATCH:-128}

cd src/diffusion

echo "=== smoke test: catches shape bugs in ~1 min before burning the allocation ==="
srun --ntasks=1 python -u smoke_test.py || { echo "SMOKE TEST FAILED - stopping"; exit 1; }

echo "=== train EDM on $ARM, target Y (bin counts) ==="
srun --ntasks=1 python -u train.py \
    --arm "$ARM" --target Y \
    --iters "$ITERS" --batch "$BATCH" --resume --preload

echo "=== evaluate on the held-out TEST split of $ARM ==="
srun --ntasks=1 python -u coverage.py \
    --ckpt "../../outputs/edm_${ARM}_Y/ckpt.pt" \
    --arms "$ARM" \
    --nsamp 256
