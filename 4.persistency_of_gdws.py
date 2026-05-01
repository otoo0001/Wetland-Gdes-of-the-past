#!/usr/bin/env python3
"""
creating_max_presence_mask.py
-----------------------------
Creates a wetGDE max-presence mask (GeoTIFF) from an already computed
climatological max NetCDF.

Classes:
    -1 - NoData / excluded
     0 - Non-GDE
     1 - GDE present
"""

import os
import math
import time
import logging
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import xarray as xr
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
from rasterio.windows import Window
from dask.diagnostics import ProgressBar

# ── Configuration ─────────────────────────────────────────────────────────────

# Historical climatology (mean)
# CLIM_NC = "/gpfs/work2/0/prjs1578/futurewetgde/wetGDEs_rechunked/wetgde_historical_clim_mean.nc"
# CLIM_VAR = "wetGDE"
# OUTPUT_DIR = "/gpfs/scratch1/shared/otoo0001/paper_2/output_gde/persistence_mask"
# LOG_FILE = "/gpfs/scratch1/shared/otoo0001/paper_2/output_gde/persistence_mask/wetgde_persistence_mask.log"

# Revised scratch paths (mean-based workflow)
# CLIM_NC = "/scratch-shared/otoo0001/past_gde/wetGDE_clim_2015_2019.nc"
# OUTPUT_DIR = "/scratch-shared/otoo0001/paper2_revisions/shapefiles"
# LOG_FILE = "/scratch-shared/otoo0001/paper2_revisions/shapefiles/logs/persistence_mask/wetgde_persistence_mask.log"

# Max-based workflow (ACTIVE)
CLIM_NC = "/scratch-shared/otoo0001/past_gde/wetGDE_clim_2015_2019_max.nc"
CLIM_VAR = "wetGDE"
OUTPUT_DIR = "/scratch-shared/otoo0001/paper2_revisions/shapefiles"
LOG_FILE = "/scratch-shared/otoo0001/paper2_revisions/shapefiles/logs/persistence_mask/wetgde_max_presence_mask.log"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)

PERIOD_START = 2015
PERIOD_END = 2019
NODATA = -1

# Optional land mask
# LAND_MASK_FILE = "/path/to/landmask.nc"
# LAND_MASK_VAR = "land"
LAND_MASK_FILE = None
LAND_MASK_VAR = None

CHUNKS = {"lat": 4096, "lon": 4096}
STRIPE_HEIGHT = 4096

# ── Logging ───────────────────────────────────────────────────────────────────

logger = logging.getLogger("wetgde_max_presence")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

if not logger.handlers:
    for h in (logging.StreamHandler(), logging.FileHandler(LOG_FILE)):
        h.setFormatter(fmt)
        logger.addHandler(h)

t0 = time.time()
logger.info("Script start")
logger.info("Opening climatology: %s", CLIM_NC)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _std(ds):
    ren = {}
    if "latitude" in ds.coords:
        ren["latitude"] = "lat"
    if "longitude" in ds.coords:
        ren["longitude"] = "lon"
    if "latitude" in ds.dims:
        ren["latitude"] = "lat"
    if "longitude" in ds.dims:
        ren["longitude"] = "lon"
    return ds.rename(ren) if ren else ds


def _res(coord):
    vals = np.asarray(coord)
    if vals.size < 2:
        raise ValueError("Coordinate has fewer than 2 values, cannot infer resolution.")
    return float(abs(vals[1] - vals[0]))


def _first_spatial_var(ds):
    candidates = []
    for v in ds.data_vars:
        dims = set(ds[v].dims)
        if {"lat", "lon"}.issubset(dims) or {"latitude", "longitude"}.issubset(dims):
            candidates.append(v)

    if len(candidates) == 0:
        raise RuntimeError(
            f"No spatial lat/lon variable found in dataset. Available data_vars: {list(ds.data_vars)}"
        )
    return candidates[0]


def _classify_block(clim_block, nodata):
    out = np.full(clim_block.shape, nodata, dtype=np.int8)
    valid = np.isfinite(clim_block)

    out[valid & (clim_block <= 0)] = 0
    out[valid & (clim_block > 0)] = 1

    return out

# ── Load climatology ──────────────────────────────────────────────────────────

ds = xr.open_dataset(CLIM_NC, decode_times=False)
ds = _std(ds)

logger.info("Dataset data variables: %s", list(ds.data_vars))
logger.info("Dataset coordinates: %s", list(ds.coords))
logger.info("Dataset dimensions: %s", dict(ds.sizes))

clim_var = CLIM_VAR if CLIM_VAR is not None else _first_spatial_var(ds)

if clim_var not in ds.data_vars:
    raise RuntimeError(
        f"Requested variable '{clim_var}' not found. Available data_vars: {list(ds.data_vars)}"
    )

clim_data = ds[clim_var].astype("float32")

for dim in list(clim_data.dims):
    if dim not in {"lat", "lon", "time"} and clim_data.sizes[dim] == 1:
        clim_data = clim_data.isel({dim: 0}, drop=True)

if "time" in clim_data.dims:
    if clim_data.sizes["time"] != 1:
        raise RuntimeError(
            f"Expected max climatology with no time dimension or singleton time, found time={clim_data.sizes['time']}"
        )
    clim_data = clim_data.isel(time=0, drop=True)

rename_dims = {}
if "latitude" in clim_data.dims:
    rename_dims["latitude"] = "lat"
if "longitude" in clim_data.dims:
    rename_dims["longitude"] = "lon"
if rename_dims:
    clim_data = clim_data.rename(rename_dims)

if not {"lat", "lon"}.issubset(set(clim_data.dims)):
    raise RuntimeError(f"Expected lat/lon dimensions, found: {clim_data.dims}")

clim_data = clim_data.chunk(CHUNKS)

lat = clim_data["lat"].values
lon = clim_data["lon"].values

if lat.ndim != 1 or lon.ndim != 1:
    raise ValueError("Expected 1D lat/lon coordinates.")

lat_res = _res(lat)
lon_res = _res(lon)
lat_desc = bool(lat[0] > lat[-1])

logger.info("Using variable: %s", clim_var)
logger.info("Input shape: lat=%d, lon=%d", clim_data.sizes["lat"], clim_data.sizes["lon"])
logger.info("Input chunks: %s", {k: tuple(v) for k, v in clim_data.chunksizes.items()})
logger.info("Latitude descending: %s", lat_desc)
logger.info("Resolution: lat=%.8f, lon=%.8f", lat_res, lon_res)

# ── Optional land mask ────────────────────────────────────────────────────────

if LAND_MASK_FILE is not None:
    logger.info("Opening land mask → %s", LAND_MASK_FILE)
    land_ds = xr.open_dataset(LAND_MASK_FILE, decode_times=False)
    land_ds = _std(land_ds)

    land_var = LAND_MASK_VAR if LAND_MASK_VAR is not None else _first_spatial_var(land_ds)

    if land_var not in land_ds.data_vars:
        raise RuntimeError(
            f"Requested land mask variable '{land_var}' not found. Available data_vars: {list(land_ds.data_vars)}"
        )

    land = land_ds[land_var]

    for dim in list(land.dims):
        if dim not in {"lat", "lon", "time"} and land.sizes[dim] == 1:
            land = land.isel({dim: 0}, drop=True)

    if "time" in land.dims:
        land = land.isel(time=0, drop=True)

    rename_dims = {}
    if "latitude" in land.dims:
        rename_dims["latitude"] = "lat"
    if "longitude" in land.dims:
        rename_dims["longitude"] = "lon"
    if rename_dims:
        land = land.rename(rename_dims)

    if not {"lat", "lon"}.issubset(set(land.dims)):
        raise RuntimeError(f"Land mask must have lat/lon dims, found: {land.dims}")

    land = land.interp(lat=clim_data.lat, lon=clim_data.lon, method="nearest")
    clim_data = clim_data.where(land == 1)
    logger.info("Applied land mask, open water excluded")
else:
    logger.info("No land mask provided, open water is not explicitly excluded")

# ── Output georeferencing ─────────────────────────────────────────────────────

west = float(lon.min() - lon_res / 2.0)
north = float(lat.max() + lat_res / 2.0)
transform = from_origin(west, north, lon_res, lat_res)

height = clim_data.sizes["lat"]
width = clim_data.sizes["lon"]

out_tif = str(Path(OUTPUT_DIR) / "wetgde_max_presence_mask_2015_2019.tif")
logger.info("Writing output → %s", out_tif)

# ── Stripe-based write ────────────────────────────────────────────────────────

n_stripes = math.ceil(height / STRIPE_HEIGHT)
class_counts = {NODATA: 0, 0: 0, 1: 0}

with rasterio.open(
    out_tif,
    "w",
    driver="GTiff",
    height=height,
    width=width,
    count=1,
    dtype="int8",
    crs=CRS.from_epsg(4326),
    transform=transform,
    compress="lzw",
    tiled=True,
    blockxsize=min(512, width),
    blockysize=min(512, height),
    nodata=NODATA,
) as dst:

    for i, row0 in enumerate(range(0, height, STRIPE_HEIGHT), start=1):
        row1 = min(row0 + STRIPE_HEIGHT, height)

        logger.info("Stripe %d/%d, rows %d:%d, starting compute()", i, n_stripes, row0, row1)
        stripe_da = clim_data.isel(lat=slice(row0, row1))

        t_stripe = time.time()
        with ProgressBar():
            stripe_np = stripe_da.compute().values

        logger.info(
            "Stripe %d/%d, rows %d:%d, compute done in %.1f s",
            i, n_stripes, row0, row1, time.time() - t_stripe
        )

        mask_np = _classify_block(stripe_np, NODATA)

        vals, counts = np.unique(mask_np, return_counts=True)
        for v, c in zip(vals, counts):
            class_counts[int(v)] += int(c)

        if lat_desc:
            write_row0 = row0
            out_block = mask_np
        else:
            write_row0 = height - row1
            out_block = np.flipud(mask_np)

        logger.info("Stripe %d/%d, rows %d:%d, writing", i, n_stripes, row0, row1)

        dst.write(
            out_block,
            1,
            window=Window(
                col_off=0,
                row_off=write_row0,
                width=width,
                height=row1 - row0,
            )
        )

        file_size_gb = Path(out_tif).stat().st_size / 1e9
        logger.info(
            "Stripe %d/%d complete, elapsed %.1f min, file size %.2f GB, class counts so far %s",
            i, n_stripes, (time.time() - t0) / 60.0, file_size_gb, class_counts
        )

    dst.update_tags(
        source_climatology=CLIM_NC,
        source_variable=clim_var,
        period=f"{PERIOD_START} to {PERIOD_END}",
        climatology_units="maximum wetGDE climatological value",
        class_nodata="No data",
        class_0="Non-GDE",
        class_1="GDE present",
    )

logger.info("Final class counts: %s", class_counts)
logger.info("Done in %.1f s", time.time() - t0)