#!/bin/bash -l
#SBATCH --job-name=pcct-gendata
#SBATCH -p Quick
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ahmedlamidi@usf.edu
#SBATCH --output=logs/gendata_%j.out
#SBATCH --error=logs/gendata_%j.err

# ONE arm, for the WGAN-vs-diffusion baseline comparison.
#
# Default = the arm that matches Morovati et al. 2025: their input is "charge
# splitting, pulse pileup, and Poisson noise", so pile-up is ON (paralyzable,
# tau = 30 ns). Both models train AND test on this arm (train / val / test split
# by phantom), so there is no mismatch in this experiment.
#
# The second arm, for the mismatch experiment later, is the same script:
#   sbatch --export=ALL,TAU=0,ARM=baseline3d_matched slurm/gen_data.sh
#
# CPU-ONLY on purpose -- no --gres=gpu, generation is numpy.
#
# --crop 1854 --nrow 32 is REQUIRED, not a tuning choice. The object spans 1768 of
# the 1854 channels; the generator's default 512-channel crop keeps 29% of it and
# NO air, and pile-up lives in air (-83%..+943% there, 2-7% through the head).
# Full width gives 690 patches/projection, ~1.74M train patches (theirs 1.84M).

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source /apps/anaconda3/etc/profile.d/conda.sh
CONDA_ENV=proj
conda activate "$CONDA_ENV"
pip install -r requirements.txt

ARM=${ARM:-baseline3d_pu_matched}
TAU=${TAU:-30}

cd src

echo "=== generating $ARM  (tau = $TAU ns, paralyzable pile-up) ==="
srun --ntasks=1 python -u generate_baseline_dataset3d.py \
    --nphantom 20 --split 70 15 15 \
    --crop 1854 --nrow 32 \
    --label clean --closed_top_bin \
    --geometry parallel --pileup_mode paralyzable \
    --tau "$TAU" --T 25 \
    --out "$ARM"

echo "=== normalisation, from $ARM's TRAIN split only ==="
cd diffusion
srun --ntasks=1 python -u norm_stats.py --arm "$ARM"
