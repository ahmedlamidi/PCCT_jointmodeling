#!/bin/bash
# ONE-TIME environment setup on HiPerGator. Run interactively on a LOGIN node
# (or an hpg-dev session), from analysis/ -- NOT as a batch job:
#
#     bash slurm/hipergator/setup_env.sh
#
# Why not pip-install inside every job, like the GAIVI template: UFRC does not
# document internet access from compute nodes, and reinstalling per job is slow.
# Build once, then every job only activates.
set -eo pipefail
# Under sbatch, $0 is Slurm's spooled copy (/var/spool/slurmd/...), not this file,
# so dirname "$0" points nowhere useful. Fall back to the submit directory.
if [ -n "$SLURM_JOB_ID" ]; then cd "$SLURM_SUBMIT_DIR"
else cd "$(dirname "$0")/../.."; fi                 # -> analysis/
[ -f requirements.txt ] || { echo "no requirements.txt in $(pwd) -- run from analysis/: bash slurm/hipergator/setup_env.sh" >&2; exit 1; }

GROUP=${GROUP:-$(id -gn)}
ENV=${PCCT_ENV:-/blue/$GROUP/$USER/conda/envs/pcct}  # /blue, not home (40 GB quota)

module load conda
eval "$(conda shell.bash hook)"
[ -d "$ENV" ] || conda create -y -p "$ENV" python=3.11
conda activate "$ENV"

# UFRC: PyTorch no longer ships through conda. The cu128 wheel covers L4
# (CUDA >= 12.0) and B200 (CUDA >= 12.8.1) GPUs.
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

python -c "import torch, numpy, scipy, h5py, matplotlib; \
print('torch', torch.__version__, '| CUDA build', torch.version.cuda)"
echo "environment ready: $ENV"
