#!/bin/bash
#SBATCH --job-name=pcct-morovati
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ahmedlamidi@usf.edu
#SBATCH --output=logs/morovati_%j.out
#SBATCH --error=logs/morovati_%j.err
#
# HiPerGator: Morovati et al. 2025's OWN phantom problem (sphere + 5 ellipsoids,
# six NIST tissues, 256^3 x 0.113 mm, 10 train + 5 test). CPU only (numpy).
# All matching choices are the script's defaults -- see its docstring.
#
# Does NOT run norm_stats.py: that writes the single outputs/norm_stats.json the
# head-arm models were trained with. Training on this arm needs its own file first.
#
#   sbatch --account=<group> --qos=<group> slurm/hipergator/gen_morovati.sh

cd "$SLURM_SUBMIT_DIR"
GROUP=${GROUP:-$(id -gn)}
ENV=${PCCT_ENV:-/blue/$GROUP/$USER/conda/envs/pcct}
module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate "$ENV"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

ARM=${ARM:-morovati_match}
TAU=${TAU:-30}
cd src

echo "=== generating $ARM  (Morovati phantom, tau = $TAU ns)  $(date) ==="
srun python -u generate_morovati_dataset.py --tau "$TAU" --out "$ARM"
