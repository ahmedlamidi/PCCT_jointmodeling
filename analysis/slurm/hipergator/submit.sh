#!/bin/bash
# Submit the whole comparison on HiPerGator, with dependencies. From analysis/:
#
#     bash slurm/hipergator/submit.sh               # investment QOS (your group)
#     QOS=burst bash slurm/hipergator/submit.sh     # burst QOS (<group>-b, 96 h cap)
#
# Run preflight.sh first.
set -eo pipefail
cd "$(dirname "$0")/../.."                          # -> analysis/
mkdir -p logs            # SLURM opens logs/*.out BEFORE the job starts: must exist now

export GROUP=${GROUP:-$(id -gn)}
case "${QOS:-investment}" in
  burst) Q="$GROUP-b" ;;
  *)     Q="$GROUP"   ;;
esac
A="--account=$GROUP --qos=$Q"

J1=$(sbatch --parsable $A slurm/hipergator/gen_data.sh)
J2=$(sbatch --parsable $A --dependency=afterok:$J1 slurm/hipergator/train_diffusion.sh)
J3=$(sbatch --parsable $A --dependency=afterok:$J1 slurm/hipergator/train_wgan.sh)

echo "submitted  gendata=$J1  edm=$J2  wgan=$J3   (account=$GROUP qos=$Q)"
echo "watch:     squeue -u $USER      logs: logs/*_<jobid>.out"
echo "when both training jobs finish:  cd src && python compare_baseline.py"
