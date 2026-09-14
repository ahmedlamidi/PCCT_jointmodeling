#!/bin/bash
#SBATCH --job-name=pcct-gendata
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ahmedlamidi@usf.edu
#SBATCH --output=logs/gendata_%j.out
#SBATCH --error=logs/gendata_%j.err
#
# HiPerGator: ONE arm for the WGAN-vs-diffusion comparison. CPU only (numpy).
# Account/QOS are passed on the command line by submit.sh, not hard-coded here.
# 24 h fits both investment (744 h) and burst (96 h) QOS; ~5-6 h expected.
#
# Default arm = pile-up ON (paralyzable, tau = 30 ns), because Morovati et al.'s
# input includes pile-up. --crop 1854 --nrow 32 is REQUIRED: the default crop keeps
# 29% of the object and no air, and pile-up lives in air.

cd "$SLURM_SUBMIT_DIR"
GROUP=${GROUP:-$(id -gn)}
ENV=${PCCT_ENV:-/blue/$GROUP/$USER/conda/envs/pcct}
module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate "$ENV"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

ARM=${ARM:-baseline3d_pu_matched}
TAU=${TAU:-30}
cd src

echo "=== generating $ARM  (tau = $TAU ns, paralyzable pile-up)  $(date) ==="
srun python -u generate_baseline_dataset3d.py \
    --nphantom 20 --split 70 15 15 \
    --crop 1854 --nrow 32 \
    --label clean --closed_top_bin \
    --geometry parallel --pileup_mode paralyzable \
    --tau "$TAU" --T 25 \
    --out "$ARM"

echo "=== normalisation from $ARM's TRAIN split  $(date) ==="
cd diffusion
srun python -u norm_stats.py --arm "$ARM"
