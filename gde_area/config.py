"""
config.py
Paths, run-time constants, and logging setup for the wetGDE area pipeline.

Set MODE = "timeseries" for the full monthly record, or "climatology" for a
12-month mean annual cycle. Everything else adapts automatically.
"""
import logging
import os
from pathlib import Path

# ── Mode ─────────────────────────────────────────────────────────────────────────
# "timeseries"  : full monthly NetCDF with a decodable or raw time axis
# "climatology" : 12-step NetCDF representing the mean annual cycle
MODE = os.environ.get("MODE", "climatology")

# ── Input files ──────────────────────────────────────────────────────────────────
WET_NC = os.environ.get(
    "WET_NC",
    "/gpfs/scratch1/shared/otoo0001/past_gde/wetGDE_clim_2015_2019.nc",
)
# Example for timeseries:
# WET_NC = os.environ.get(
#     "WET_NC",
#     "/gpfs/scratch1/shared/otoo0001/past_gde/wetGDE_40.nc",
# )

BIOME_SHP = os.environ.get(
    "BIOME_SHP",
    "/gpfs/work2/0/prjs1578/futurewetgde/shapefiles/biomes_new/biomes/wwf_terr_ecos.shp",
)

COUNTRY_SHP = os.environ.get(
    "COUNTRY_SHP",
    "/gpfs/scratch1/shared/otoo0001/data/shapefiles/"
    "WB_countries_Admin0_10m/World_Countries_Generalized.shp",
)

OPEN_WATER_TIF = os.environ.get(
    "OPEN_WATER_TIF",
    "/gpfs/scratch1/shared/otoo0001/glwd_rebuilt_masks/glwd_open_water_mask.tif",
)

PERSISTENCE_TIF = os.environ.get(
    "PERSISTENCE_TIF",
    "/gpfs/scratch1/shared/otoo0001/paper_2/output_gde/"
    "persistence_mask/wetgde_persistence_mask_historical.tif",
)

QA_FILES = [
    "/gpfs/work2/0/prjs1578/futurewetgde/quality_flags/static_quality_flags.nc",
    "/gpfs/work2/0/prjs1578/futurewetgde/quality_flags/spin_up_1960.nc",
]

# ── Cell area ────────────────────────────────────────────────────────────────────
# 30-arcsec cell area file (mean ~546,622 m² = 0.547 km² per cell).
# Variable name in this file is "cell_area", units are m².
# utils.load_cell_area() converts m² → km² automatically when CELL_AREA_UNITS="m2".
CELL_AREA_FILE = os.environ.get(
    "CELL_AREA_FILE",
    "/gpfs/scratch1/shared/otoo0001/paper_2_revisions/shapefiles/"
    "cdo_grid_area_30sec_map_correct_lat.nc",
)
CELL_AREA_VAR   = os.environ.get("CELL_AREA_VAR",   "cell_area")
CELL_AREA_UNITS = os.environ.get("CELL_AREA_UNITS", "m2")   # "m2" or "km2"

# ── Output files ─────────────────────────────────────────────────────────────────
OUT_DIR = Path(
    os.environ.get(
        "OUT_DIR",
        "/scratch-shared/otoo0001/paper2_revisions/output_area",
    )
)

OUT_BIOME_FILE        = OUT_DIR / f"gde_area_by_biome_{MODE}.parquet"
OUT_BIOME_REALM_FILE  = OUT_DIR / f"gde_area_by_biome_realm_{MODE}.parquet"
OUT_BIOME_COUNTRY_FILE= OUT_DIR / f"gde_area_by_biome_country_{MODE}.parquet"
LOG_FILE              = OUT_DIR / f"gde_area_{MODE}.log"

# ── Time series settings ─────────────────────────────────────────────────────────
START_DATE = os.environ.get("START_DATE", "1979-01-31")

# ── Run-time constants ───────────────────────────────────────────────────────────
MAX_WORKERS      = int(os.environ.get("MAX_WORKERS", 12))
TEST_MODE        = os.environ.get("TEST_MODE", "False").lower() == "true"
COUNTRY_ID_FIELD = os.environ.get("COUNTRY_ID_FIELD", "ISO")
BIOME_ID_FIELD   = os.environ.get("BIOME_ID_FIELD",   "BIOME")
REALM_ID_FIELD   = os.environ.get("REALM_ID_FIELD",   "REALM")

# ── Logging ──────────────────────────────────────────────────────────────────────
def get_logger(name: str = "gde_area") -> logging.Logger:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.StreamHandler(), logging.FileHandler(LOG_FILE)):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger