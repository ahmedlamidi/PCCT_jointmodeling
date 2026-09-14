#!/bin/bash
# Run on a HiPerGator LOGIN node, from analysis/, BEFORE submitting:
#
#     bash slurm/hipergator/preflight.sh
#
# Checks everything that can be checked without spending an allocation, and ends
# with `sbatch --test-only` on each job: the scheduler validates the partition,
# account, QOS and resources, then runs nothing.
cd "$(dirname "$0")/../.."                          # -> analysis/
GROUP=${GROUP:-$(id -gn)}
ENV=${PCCT_ENV:-/blue/$GROUP/$USER/conda/envs/pcct}
fail=0
ok()   { echo "  OK    $*"; }
warn() { echo "  WARN  $*"; }
bad()  { echo "  FAIL  $*"; fail=1; }

echo "== location =="
case "$(pwd -P)" in
  /blue/*) ok "repo is on /blue ($(pwd -P))" ;;
  *)       bad "repo is at $(pwd -P). Move it under /blue/$GROUP/$USER -- home is 40 GB and this writes ~20 GB" ;;
esac

echo "== gitignored inputs (git will NOT carry these -- rsync them) =="
for f in ../PcTK_3.24a/1_inputdata/pmf_S0_Al2.0_120kVp.mat \
         ../PcTK_3.24a/1_inputdata/m2_mukE.csv \
         ../PcTK_3.24a/5_refdata/dat_nCovE_ver3.2_dpix_225_dz_1600_r0_24_esig_2.0.mat; do
  [ -f "$f" ] && ok "$f" || bad "missing $f"
done
for f in cache/diagE.npy cache/q_sym.npy cache/Pr3.npy \
         cache/ncovw_20_30_40_50_60_70_80_90_100_110.npz; do
  [ -f "$f" ] && ok "$f" || warn "missing $f -- will be rebuilt from PcTK on first run (~10 min)"
done

echo "== account / QOS  (group assumed = $GROUP; override with GROUP=...) =="
if module load ufrc 2>/dev/null; then slurmInfo "$GROUP" 2>&1 | head -15; else warn "ufrc module not found"; fi

echo "== environment ($ENV) =="
module load conda 2>/dev/null && eval "$(conda shell.bash hook)"
if conda activate "$ENV" 2>/dev/null; then
  python -c "import torch, numpy, scipy, h5py, matplotlib; \
print('  OK    torch', torch.__version__, '| CUDA build', torch.version.cuda)" \
    || bad "imports failed in $ENV"
else
  bad "no environment at $ENV -- run: bash slurm/hipergator/setup_env.sh"
fi

echo "== scheduler dry run (sbatch --test-only: validates, runs nothing) =="
mkdir -p logs
# without this, a missing sbatch prints 'command not found' -- which matches none
# of the error words below, so every job would have been reported OK
if ! command -v sbatch >/dev/null 2>&1; then
  bad "sbatch not found -- run this on a HiPerGator login node"
else
for s in gen_data train_diffusion train_wgan; do
  out=$(sbatch --test-only --account="$GROUP" --qos="$GROUP" "slurm/hipergator/$s.sh" 2>&1)
  if echo "$out" | grep -qiE "error|invalid|denied|not found"; then bad "$s: $out"; else ok "$s: $out"; fi
done
fi

echo "== quota =="
# `cmd | head` returns head's status, so '|| warn' could never fire; test first
if command -v blue_quota >/dev/null 2>&1; then blue_quota | head -6
else warn "blue_quota not available (not on HiPerGator?)"; fi

echo
if [ $fail -eq 0 ]; then echo "ALL CHECKS PASSED  ->  bash slurm/hipergator/submit.sh"
else echo "FIX THE FAIL LINES ABOVE BEFORE SUBMITTING"; exit 1; fi
