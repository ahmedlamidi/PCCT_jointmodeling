#!/bin/bash
#SBATCH --job-name=pcct-ssim
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ahmedlamidi@usf.edu
#SBATCH --output=logs/ssim_%j.out
#SBATCH --error=logs/ssim_%j.err
#
# Windowed per-bin SSIM of the diffusion posterior mean, on test patches and on
# FBP slices, plus the sinogram error figure (src/diffusion/ssim_eval.py).
# Needs the trained checkpoint: run after train_diffusion.sh. From analysis/:
#
#     sbatch --account=$GROUP --qos=$GROUP slurm/hipergator/ssim_diffusion.sh
#
# The image domain dominates the time: 3 phantoms x 41,580 patches x NSAMP_RECON
# samples. The log prints an ETA after the first chunk.
#
# First a PREVIEW: phantom 0 with 2 samples per patch (~1/8 of its full time) ->
# figures/preview/ and outputs/ssim_..._preview/. A quick look only -- not a result,
# no uncertainty band, never read by compare_baseline.py.
#
# Then, earliest result first: phantom 0's slice -> its sinogram figure and
# time-series profiles are written as soon as that ONE phantom is done; then the
# patch domain; then phantoms 1 and 2. The JSON is rewritten after each step
# ("complete": false until the last), and each finished phantom's slice is saved,
# so on TIMEOUT resubmit the same line and it continues.
#
# The WGAN, scored the same way (one pass per patch, much faster):
#
#     sbatch --account=$GROUP --qos=$GROUP --export=ALL,MODEL=wgan slurm/hipergator/ssim_diffusion.sh

cd "$SLURM_SUBMIT_DIR"
GROUP=${GROUP:-$(id -gn)}
ENV=${PCCT_ENV:-/blue/$GROUP/$USER/conda/envs/pcct}
module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate "$ENV"
nvidia-smi -L || true

ARM=${ARM:-baseline3d_pu_matched}
MODEL=${MODEL:-edm}                 # edm = diffusion; wgan = the baseline (nsamp forced to 1)
NSAMP_RECON=${NSAMP_RECON:-16}
PREVIEW_NSAMP=${PREVIEW_NSAMP:-2}   # fast first look at phantom 0 -> figures/preview/; 0 = off
cd src/diffusion
srun python -u ssim_eval.py --model "$MODEL" --train_arm "$ARM" --nsamp 256 --nsamp_recon "$NSAMP_RECON" \
    --preview_nsamp "$PREVIEW_NSAMP"
