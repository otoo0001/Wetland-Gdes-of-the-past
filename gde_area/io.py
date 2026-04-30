"""
io.py
Input/output operations for the wetGDE area pipeline:
  - rasterize biome, realm, and country shapefiles onto the QA grid
  - save result DataFrames to parquet
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio.features
from affine import Affine

from .config import BIOME_ID_FIELD, COUNTRY_ID_FIELD, REALM_ID_FIELD

logger = logging.getLogger("gde_area")


# ── Transform helper ─────────────────────────────────────────────────────────────
def _grid_transform(lat: np.ndarray, lon: np.ndarray) -> Affine:
    lat_res = float(abs(lat[1] - lat[0]))
    lon_res = float(abs(lon[1] - lon[0]))
    return Affine(
        lon_res, 0, float(lon.min()) - lon_res / 2.0,
        0, -lat_res, float(lat.max()) + lat_res / 2.0,
    )


# ── Biome rasterization ──────────────────────────────────────────────────────────
def rasterize_biomes(
    shp_path: str,
    lat: np.ndarray,
    lon: np.ndarray,
    analysis_mask: np.ndarray,
) -> tuple[np.ndarray, list[int]]:
    """
    Rasterize WWF biomes onto the analysis grid.

    Antarctica and non-standard biome classes are excluded by keeping only
    biome IDs 1-14.

    Returns
    -------
    biome_arr : int16 array (ny, nx), 0 outside analysis mask
    biome_ids : sorted list of biome IDs present in the source layer
    """
    logger.info("Loading biome shapefile → %s", shp_path)
    gdf = gpd.read_file(shp_path)

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    else:
        gdf = gdf.to_crs("EPSG:4326")

    if BIOME_ID_FIELD not in gdf.columns:
        raise RuntimeError(
            f"Field '{BIOME_ID_FIELD}' not found. Available: {list(gdf.columns)}"
        )

    gdf = (
        gdf[[BIOME_ID_FIELD, "geometry"]]
        .dropna(subset=[BIOME_ID_FIELD, "geometry"])
        .rename(columns={BIOME_ID_FIELD: "biome_id"})
        .copy()
    )
    gdf["biome_id"] = pd.to_numeric(gdf["biome_id"], errors="coerce")
    gdf = gdf[gdf["biome_id"].between(1, 14)].copy()
    gdf["biome_id"] = gdf["biome_id"].astype(np.int16)

    biome_ids = sorted(gdf["biome_id"].unique().tolist())
    logger.info("Unique BIOME IDs kept: %s", biome_ids)

    shapes = ((geom, int(bid)) for geom, bid in zip(gdf.geometry, gdf["biome_id"]))
    biome_arr = rasterio.features.rasterize(
        shapes=shapes,
        out_shape=(len(lat), len(lon)),
        transform=_grid_transform(lat, lon),
        fill=0,
        dtype="int16",
    )
    biome_arr = np.where(analysis_mask, biome_arr, 0).astype(np.int16)

    logger.info("Rasterized biome array, shape: %s", biome_arr.shape)
    return biome_arr, biome_ids


# ── Realm rasterization ──────────────────────────────────────────────────────────
def rasterize_realms(
    shp_path: str,
    lat: np.ndarray,
    lon: np.ndarray,
    analysis_mask: np.ndarray,
) -> tuple[np.ndarray, dict[int, str]]:
    """
    Rasterize realms onto the analysis grid.

    Antarctica is excluded by dropping realm code 'AN'.

    Returns
    -------
    realm_arr : int16 array (ny, nx), 0 outside analysis mask
    code_to_realm : dict mapping integer code to realm string
    """
    logger.info("Loading realm shapefile → %s", shp_path)
    gdf = gpd.read_file(shp_path)

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    else:
        gdf = gdf.to_crs("EPSG:4326")

    if REALM_ID_FIELD not in gdf.columns:
        raise RuntimeError(
            f"Field '{REALM_ID_FIELD}' not found. Available: {list(gdf.columns)}"
        )

    gdf = (
        gdf[[REALM_ID_FIELD, "geometry"]]
        .dropna(subset=[REALM_ID_FIELD, "geometry"])
        .copy()
    )
    gdf[REALM_ID_FIELD] = gdf[REALM_ID_FIELD].astype(str).str.strip()
    gdf = gdf[
        gdf[REALM_ID_FIELD].notna()
        & (gdf[REALM_ID_FIELD] != "")
        & (gdf[REALM_ID_FIELD].str.lower() != "nan")
        & (gdf[REALM_ID_FIELD] != "AN")
    ].copy()

    unique_realms = sorted(gdf[REALM_ID_FIELD].unique().tolist())
    realm_to_code = {realm: i + 1 for i, realm in enumerate(unique_realms)}
    code_to_realm = {code: realm for realm, code in realm_to_code.items()}

    shapes = (
        (geom, realm_to_code[realm])
        for geom, realm in zip(gdf.geometry, gdf[REALM_ID_FIELD])
    )
    realm_arr = rasterio.features.rasterize(
        shapes=shapes,
        out_shape=(len(lat), len(lon)),
        transform=_grid_transform(lat, lon),
        fill=0,
        dtype="int16",
    )
    realm_arr = np.where(analysis_mask, realm_arr, 0).astype(np.int16)

    logger.info("Rasterized realm array, shape: %s", realm_arr.shape)
    return realm_arr, code_to_realm


# ── Country rasterization ────────────────────────────────────────────────────────
def rasterize_countries(
    shp_path: str,
    lat: np.ndarray,
    lon: np.ndarray,
    analysis_mask: np.ndarray,
) -> tuple[np.ndarray, dict[int, str]]:
    """
    Rasterize countries onto the analysis grid using ISO codes.

    Antarctica is excluded by dropping ISO code 'ATA'.

    Returns
    -------
    country_arr : int32 array (ny, nx), 0 outside analysis mask
    code_to_country : dict mapping integer code to ISO string
    """
    logger.info("Loading country shapefile → %s", shp_path)
    gdf = gpd.read_file(shp_path)

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    else:
        gdf = gdf.to_crs("EPSG:4326")

    if COUNTRY_ID_FIELD not in gdf.columns:
        raise RuntimeError(
            f"Field '{COUNTRY_ID_FIELD}' not found. Available: {list(gdf.columns)}"
        )

    gdf = (
        gdf[[COUNTRY_ID_FIELD, "geometry"]]
        .dropna(subset=[COUNTRY_ID_FIELD, "geometry"])
        .copy()
    )
    gdf[COUNTRY_ID_FIELD] = gdf[COUNTRY_ID_FIELD].astype(str).str.strip()
    gdf = gdf[
        gdf[COUNTRY_ID_FIELD].notna()
        & (gdf[COUNTRY_ID_FIELD] != "")
        & (gdf[COUNTRY_ID_FIELD] != "-99")
        & (gdf[COUNTRY_ID_FIELD].str.lower() != "nan")
        & (gdf[COUNTRY_ID_FIELD] != "ATA")
    ].copy()

    unique_countries = sorted(gdf[COUNTRY_ID_FIELD].unique().tolist())
    country_to_code = {cid: i + 1 for i, cid in enumerate(unique_countries)}
    code_to_country = {code: cid for cid, code in country_to_code.items()}

    shapes = (
        (geom, country_to_code[cid])
        for geom, cid in zip(gdf.geometry, gdf[COUNTRY_ID_FIELD])
    )
    country_arr = rasterio.features.rasterize(
        shapes=shapes,
        out_shape=(len(lat), len(lon)),
        transform=_grid_transform(lat, lon),
        fill=0,
        dtype="int32",
    )
    country_arr = np.where(analysis_mask, country_arr, 0).astype(np.int32)

    logger.info("Rasterized country array, shape: %s", country_arr.shape)
    return country_arr, code_to_country


# ── Save helpers ─────────────────────────────────────────────────────────────────
def save_results(
    biome_results: list[dict],
    biome_realm_results: list[dict],
    biome_country_results: list[dict],
    out_biome_file: Path,
    out_biome_realm_file: Path,
    out_biome_country_file: Path,
    time_col: str,
) -> None:
    """
    Sort and write biome, biome-realm, and biome-country result lists to parquet.
    """
    biome_df = pd.DataFrame(biome_results)
    biome_realm_df = pd.DataFrame(biome_realm_results)
    biome_country_df = pd.DataFrame(biome_country_results)

    if biome_df.empty:
        biome_df = pd.DataFrame(columns=[time_col, "biome_id", "area_km2"])
    else:
        biome_df = (
            biome_df.sort_values([time_col, "biome_id"])
            .reset_index(drop=True)
        )

    if biome_realm_df.empty:
        biome_realm_df = pd.DataFrame(columns=[time_col, "biome_id", "realm", "area_km2"])
    else:
        biome_realm_df = (
            biome_realm_df.sort_values([time_col, "biome_id", "realm"])
            .reset_index(drop=True)
        )

    if biome_country_df.empty:
        biome_country_df = pd.DataFrame(
            columns=[time_col, "biome_id", COUNTRY_ID_FIELD, "area_km2"]
        )
    else:
        biome_country_df = (
            biome_country_df.sort_values([time_col, "biome_id", COUNTRY_ID_FIELD])
            .reset_index(drop=True)
        )

    out_biome_file.parent.mkdir(parents=True, exist_ok=True)
    out_biome_realm_file.parent.mkdir(parents=True, exist_ok=True)
    out_biome_country_file.parent.mkdir(parents=True, exist_ok=True)

    biome_df.to_parquet(out_biome_file, index=False)
    biome_realm_df.to_parquet(out_biome_realm_file, index=False)
    biome_country_df.to_parquet(out_biome_country_file, index=False)

    logger.info("Wrote %d biome rows → %s", len(biome_df), out_biome_file)
    logger.info("Wrote %d biome-realm rows → %s", len(biome_realm_df), out_biome_realm_file)
    logger.info("Wrote %d biome-country rows → %s", len(biome_country_df), out_biome_country_file)