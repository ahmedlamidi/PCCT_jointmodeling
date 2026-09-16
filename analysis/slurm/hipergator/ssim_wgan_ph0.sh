#!/bin/bash
#SBATCH --job-name=pcct-ssim-wgan
#SBATCH --account=sheng1.usf
#SBATCH --qos=sheng1.usf
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ahmedlamidi@usf.edu
#SBATCH --output=logs/ssim_wgan_%j.out
#SBATCH --error=logs/ssim_wgan_%j.err
#
# WGAN on the SAME slice the diffusion maps use: test phantom 0, detector row 16,
# plus the 256 test patches. No preview. Run after train_wgan.sh has finished.
# From analysis/:
#
#     sbatch slurm/hipergator/ssim_wgan_ph0.sh
#
# Writes outputs/ssim_wgan_<arm>_Y_on_<arm>/slice_ph000_row16.npz (same layout as
# the diffusion one; std = 0). Phantoms 1-2 for the full table: ssim_diffusion.sh
# with MODEL=wgan.

cd "$SLURM_SUBMIT_DIR"
ENV=${PCCT_ENV:-/blue/sheng1.usf/$USER/conda/envs/pcct}
module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate "$ENV"
nvidia-smi -L || true

cd src/diffusion
srun python -u ssim_eval.py --model wgan --train_arm baseline3d_pu_matched \
    --phantoms 0 --preview_nsamp 0
