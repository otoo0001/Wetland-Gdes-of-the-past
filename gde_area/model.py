"""
model.py
Main orchestration for the wetGDE area pipeline.

Supports two modes, set via config.MODE:
  "timeseries"  : full monthly record, time labels are pandas Timestamps
  "climatology" : 12-step mean annual cycle, time labels are integers 1-12

Run with:
    python -m gde_area.model

Outputs:
    gde_area_by_biome_{mode}.parquet
    gde_area_by_biome_realm_{mode}.parquet
    gde_area_by_biome_country_{mode}.parquet
"""
import time

import numpy as np
import pandas as pd
import xarray as xr

from .config import (
    WET_NC,
    BIOME_SHP,
    COUNTRY_SHP,
    OPEN_WATER_TIF,
    PERSISTENCE_TIF,
    QA_FILES,
    OUT_BIOME_FILE,
    OUT_BIOME_REALM_FILE,
    OUT_BIOME_COUNTRY_FILE,
    MODE,
    START_DATE,
    TEST_MODE,
    COUNTRY_ID_FIELD,
    get_logger,
)
from .utils import (
    _std,
    open_qa_merged,
    build_qa_mask,
    load_open_water_mask,
    load_persistence_mask,
    load_cell_area,
)
from .io import (
    rasterize_biomes,
    rasterize_realms,
    rasterize_countries,
    save_results,
)


def _assign_time_labels(wet_ds: xr.Dataset) -> tuple[xr.Dataset, list, str]:
    """
    Assign interpretable time labels to the wetGDE dataset and return the
    labels plus the output time-column name.
    """
    n = wet_ds.sizes["time"]

    if MODE == "climatology":
        if n != 12:
            raise ValueError(
                f"MODE='climatology' expects 12 time steps, found {n}. "
                "Check WET_NC or set MODE='timeseries'."
            )
        wet_ds = wet_ds.assign_coords(time=np.arange(1, 13))
        return wet_ds, list(wet_ds["time"].values), "month"

    try:
        decoded = xr.open_dataset(WET_NC)
        decoded = _std(decoded)
        labels = list(pd.to_datetime(decoded["time"].values))
        decoded.close()
    except Exception:
        labels = list(pd.date_range(START_DATE, periods=n, freq="ME"))

    wet_ds = wet_ds.assign_coords(time=np.arange(n))
    return wet_ds, labels, "time"


def _aggregate_1d_sum(
    codes: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sum weights by a single integer code array.
    Code 0 is treated as background and ignored.
    """
    valid = (codes > 0) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return np.array([], dtype=int), np.array([], dtype=float)

    sums = np.bincount(codes[valid], weights=weights[valid])
    ids = np.nonzero(sums > 0)[0]
    ids = ids[ids > 0]

    return ids.astype(int), sums[ids].astype(float)


def _aggregate_2d_sum(
    code1: np.ndarray,
    code2: np.ndarray,
    weights: np.ndarray,
    n_code2: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sum weights by all combinations of two integer code arrays.
    Code 0 in either array is treated as background and ignored.
    """
    valid = (code1 > 0) & (code2 > 0) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return (
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.array([], dtype=float),
        )

    combo = code1[valid].astype(np.int64) * (n_code2 + 1) + code2[valid].astype(np.int64)
    sums = np.bincount(combo, weights=weights[valid])

    idx = np.nonzero(sums > 0)[0]
    idx = idx[idx > 0]

    out1 = idx // (n_code2 + 1)
    out2 = idx % (n_code2 + 1)
    outw = sums[idx]

    keep = (out1 > 0) & (out2 > 0) & (outw > 0)
    return out1[keep].astype(int), out2[keep].astype(int), outw[keep].astype(float)


def run():
    logger = get_logger()
    t0 = time.time()
    logger.info("Pipeline start, MODE = %s", MODE)

    # ── QA flags ────────────────────────────────────────────────────────────────
    logger.info("Loading QA flags")
    qa_ds = open_qa_merged(QA_FILES)
    qa_lat = qa_ds["lat"].values
    qa_lon = qa_ds["lon"].values

    logger.info("Loading open-water mask → %s", OPEN_WATER_TIF)
    open_water_mask = load_open_water_mask(OPEN_WATER_TIF, qa_lat, qa_lon)

    qa_mask = build_qa_mask(qa_ds, open_water_mask)
    logger.info(
        "QA mask: %.1f%% valid, %.1f%% open-water excluded",
        100.0 * qa_mask.mean(),
        100.0 * open_water_mask.mean(),
    )
    qa_ds.close()

    # ── Persistence mask ────────────────────────────────────────────────────────
    logger.info("Loading persistence mask → %s", PERSISTENCE_TIF)
    persistence_arr = load_persistence_mask(PERSISTENCE_TIF, qa_lat, qa_lon)
    gde_persistence_mask = persistence_arr > 0
    logger.info(
        "Persistence footprint: %.1f%% of cells",
        100.0 * gde_persistence_mask.mean(),
    )

    # ── Combined analysis mask ──────────────────────────────────────────────────
    analysis_mask = qa_mask & gde_persistence_mask
    logger.info("Analysis mask: %.1f%% of cells", 100.0 * analysis_mask.mean())

    # ── Cell area ───────────────────────────────────────────────────────────────
    cell_area = load_cell_area(qa_lat, qa_lon)
    logger.info("Cell area array shape: %s", cell_area.shape)
    logger.info(
        "Total grid area on analysis grid: %.3f km²",
        float(np.nansum(cell_area)),
    )

    # ── Rasterized zone codes ───────────────────────────────────────────────────
    # Antarctica should already be excluded inside io.py:
    # - biome ids restricted to 1–14
    # - realm code/name "AN" excluded
    # - country code "ATA" excluded
    biome_arr, biome_ids = rasterize_biomes(BIOME_SHP, qa_lat, qa_lon, analysis_mask)
    realm_arr, code_to_realm = rasterize_realms(BIOME_SHP, qa_lat, qa_lon, analysis_mask)
    country_arr, code_to_country = rasterize_countries(
        COUNTRY_SHP, qa_lat, qa_lon, analysis_mask
    )

    logger.info("Biome count: %d", len(biome_ids))
    logger.info("Realm count: %d", len(code_to_realm))
    logger.info("Country count: %d", len(code_to_country))

    # ── Open wetGDE dataset ─────────────────────────────────────────────────────
    logger.info("Opening wetGDE → %s", WET_NC)
    wet_ds = _std(xr.open_dataset(WET_NC, decode_times=False))

    if "wetGDE" not in wet_ds.data_vars:
        raise RuntimeError(
            f"'wetGDE' not found in {WET_NC}. Found: {list(wet_ds.data_vars)}"
        )

    wet_ds, time_labels, time_col = _assign_time_labels(wet_ds)
    logger.info(
        "Time axis: %d steps, col = '%s', first = %s, last = %s",
        len(time_labels),
        time_col,
        time_labels[0],
        time_labels[-1],
    )

    if TEST_MODE:
        time_labels = time_labels[:1]
        logger.info("TEST MODE: processing first step only (%s)", time_labels[0])

    # ── Pre-flatten static arrays once ──────────────────────────────────────────
    biome_flat = biome_arr.ravel()
    realm_flat = realm_arr.ravel()
    country_flat = country_arr.ravel()
    analysis_flat = analysis_mask.ravel()
    cell_area_flat = cell_area.ravel()

    n_realm = max(code_to_realm.keys()) if code_to_realm else 0
    n_country = max(code_to_country.keys()) if code_to_country else 0

    # ── Main loop ───────────────────────────────────────────────────────────────
    biome_results = []
    biome_realm_results = []
    biome_country_results = []

    for i, label in enumerate(time_labels):
        logger.info("[%d/%d] Processing %s", i + 1, len(time_labels), label)

        wet_t = (
            wet_ds["wetGDE"]
            .isel(time=i)
            .interp(lat=qa_lat, lon=qa_lon, method="nearest")
        )

        wet_vals = np.nan_to_num(wet_t.values, nan=0.0)
        wet_vals[~analysis_mask] = 0.0

        wet_area_flat = wet_vals.ravel() * cell_area_flat
        wet_area_flat[~analysis_flat] = 0.0

        # biome
        b_ids, b_area = _aggregate_1d_sum(biome_flat, wet_area_flat)
        for bid, area in zip(b_ids, b_area):
            biome_results.append(
                {
                    time_col: label,
                    "biome_id": int(bid),
                    "area_km2": float(area),
                }
            )

        # biome × realm
        br_b, br_r, br_a = _aggregate_2d_sum(
            biome_flat,
            realm_flat,
            wet_area_flat,
            n_realm,
        )
        for bid, rcode, area in zip(br_b, br_r, br_a):
            biome_realm_results.append(
                {
                    time_col: label,
                    "biome_id": int(bid),
                    "realm": code_to_realm[int(rcode)],
                    "area_km2": float(area),
                }
            )

        # biome × country
        bc_b, bc_c, bc_a = _aggregate_2d_sum(
            biome_flat,
            country_flat,
            wet_area_flat,
            n_country,
        )
        for bid, ccode, area in zip(bc_b, bc_c, bc_a):
            biome_country_results.append(
                {
                    time_col: label,
                    "biome_id": int(bid),
                    COUNTRY_ID_FIELD: code_to_country[int(ccode)],
                    "area_km2": float(area),
                }
            )

    wet_ds.close()

    logger.info("Biome result rows: %d", len(biome_results))
    logger.info("Biome-realm result rows: %d", len(biome_realm_results))
    logger.info("Biome-country result rows: %d", len(biome_country_results))

    # ── Save ────────────────────────────────────────────────────────────────────
    save_results(
        biome_results=biome_results,
        biome_realm_results=biome_realm_results,
        biome_country_results=biome_country_results,
        out_biome_file=OUT_BIOME_FILE,
        out_biome_realm_file=OUT_BIOME_REALM_FILE,
        out_biome_country_file=OUT_BIOME_COUNTRY_FILE,
        time_col=time_col,
    )

    logger.info("Pipeline complete in %.1f s", time.time() - t0)


if __name__ == "__main__":
    run()