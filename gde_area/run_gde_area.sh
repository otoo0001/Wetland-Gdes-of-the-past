#!/bin/bash
#SBATCH --job-name=wetgde_area_pipeline
#SBATCH --partition=fat_genoa
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nicholetylor@gmail.com
#SBATCH --output=/gpfs/scratch1/shared/otoo0001/paper_2/output_gde/logs/%x_%j.out
#SBATCH --error=/gpfs/scratch1/shared/otoo0001/paper_2/output_gde/logs/%x_%j.err
#SBATCH --export=NONE

set -euo pipefail
set +u; . "/home/otoo0001/load_all_default.sh"; set -u

echo "[run] $(date -Iseconds)"

# ── Python env ────────────────────────────────────────────────────────────────
PYENV="/home/otoo0001/miniconda3/envs/gdes_area"
PYBIN="${PYENV}/bin/python"
if [ ! -x "${PYBIN}" ]; then
    echo "[error] Python not found/executable: ${PYBIN}"
    exit 2
fi
echo "[env] python=${PYBIN}"
"${PYBIN}" -V

# sanity check key packages
"${PYBIN}" - <<'EOF'
import xarray, numpy, geopandas, rasterio, pandas
print(f"xarray={xarray.__version__} numpy={numpy.__version__} pandas={pandas.__version__}")
print(f"geopandas={geopandas.__version__} rasterio={rasterio.__version__}")
EOF

# ── paths ─────────────────────────────────────────────────────────────────────
export OUT_DIR="/scratch-shared/otoo0001/paper2_revisions/output_area"
export LOG_DIR="${OUT_DIR}/logs"
mkdir -p "$LOG_DIR"

# ── stability ─────────────────────────────────────────────────────────────────
export TERM=dumb
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
mkdir -p "$TMPDIR"

# parent directory containing the gde_area package folder
PACKAGE_DIR="/home/otoo0001/github/paper_2/"

# ── run ───────────────────────────────────────────────────────────────────────
echo "[run] $(date -Iseconds) starting gde_area pipeline"
cd "$PACKAGE_DIR"
/usr/bin/time -v "${PYBIN}" -u -X faulthandler -m gde_area.model
echo "[done] $(date -Iseconds)"