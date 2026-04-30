"""
utils.py
Low-level helpers for the wetGDE area pipeline:
  - coordinate standardisation
  - QA flag loading and merging
  - QA mask construction
  - open-water and persistence mask reprojection
  - cell-area loading and alignment
  - per-mask area summation
"""
from pathlib import Path

import numpy as np
import xarray as xr
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from affine import Affine

from .config import CELL_AREA_FILE, CELL_AREA_UNITS, CELL_AREA_VAR


# ── Coordinate standardisation ───────────────────────────────────────────────────
def _std(ds: xr.Dataset) -> xr.Dataset:
    """Rename latitude/longitude to lat/lon if needed."""
    ren = {}
    if "latitude" in ds.coords:
        ren["latitude"] = "lat"
    if "longitude" in ds.coords:
        ren["longitude"] = "lon"
    return ds.rename(ren) if ren else ds


# ── QA loading ───────────────────────────────────────────────────────────────────
def open_qa_merged(qa_paths: list[str]) -> xr.Dataset:
    """
    Open and merge QA flag files onto a common grid.
    Variables with a time dimension are collapsed to their first slice.
    Mismatched grids are aligned with nearest-neighbour interpolation.
    """
    base = _std(xr.open_dataset(qa_paths[0], decode_times=False, mask_and_scale=False))
    qa_lat = base["lat"].values
    qa_lon = base["lon"].values
    ny0, nx0 = qa_lat.size, qa_lon.size

    qa_vars = {}
    for p in qa_paths:
        ds = _std(xr.open_dataset(p, decode_times=False, mask_and_scale=False))
        grid_mismatch = (
            ds.sizes.get("lat") != ny0
            or ds.sizes.get("lon") != nx0
            or not np.array_equal(ds["lat"].values, qa_lat)
            or not np.array_equal(ds["lon"].values, qa_lon)
        )
        if grid_mismatch:
            ds = ds.interp(lat=qa_lat, lon=qa_lon, method="nearest")
        for v in ds.data_vars:
            dv = ds[v]
            if "time" in dv.dims:
                dv = dv.isel(time=0, drop=True)
            qa_vars[v] = dv
        ds.close()

    base.close()
    return xr.Dataset(qa_vars, coords={"lat": qa_lat, "lon": qa_lon})


# ── QA mask builder ──────────────────────────────────────────────────────────────
def build_qa_mask(qa_ds: xr.Dataset, open_water_mask: np.ndarray) -> np.ndarray:
    """
    Combine static QA flags, spin-up flag, and open-water mask into a single
    boolean keep-mask, True means valid cell.
    """
    ny, nx = len(qa_ds["lat"]), len(qa_ds["lon"])
    static_ok = np.ones((ny, nx), dtype=bool)

    for flag in ("mountains_qa", "karst_qa", "permafrost_qa"):
        if flag in qa_ds.data_vars:
            static_ok &= (qa_ds[flag].values == 0)

    spin = qa_ds["spinup_qa"].values if "spinup_qa" in qa_ds.data_vars else None
    spin_ok = ~np.isin(spin, [4, 6]) if spin is not None else np.ones((ny, nx), dtype=bool)

    return static_ok & spin_ok & (~open_water_mask)


# ── Raster reprojection helpers ──────────────────────────────────────────────────
def _dst_transform(lat: np.ndarray, lon: np.ndarray) -> Affine:
    lat_res = float(abs(lat[1] - lat[0]))
    lon_res = float(abs(lon[1] - lon[0]))
    return Affine(
        lon_res, 0, float(lon.min()) - lon_res / 2.0,
        0, -lat_res, float(lat.max()) + lat_res / 2.0,
    )


def load_open_water_mask(mask_tif: str, qa_lat: np.ndarray, qa_lon: np.ndarray) -> np.ndarray:
    """
    Reproject open-water GeoTIFF onto the QA grid.
    Returns boolean array, True where open water is present.
    """
    if not Path(mask_tif).is_file():
        raise FileNotFoundError(f"Open-water mask not found: {mask_tif}")

    dst = np.zeros((len(qa_lat), len(qa_lon)), dtype=np.float32)

    with rasterio.open(mask_tif) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=_dst_transform(qa_lat, qa_lon),
            dst_crs="EPSG:4326",
            dst_nodata=0,
            resampling=Resampling.nearest,
        )

    return dst > 0


def load_persistence_mask(mask_tif: str, tgt_lat: np.ndarray, tgt_lon: np.ndarray) -> np.ndarray:
    """
    Reproject persistence mask GeoTIFF onto the target grid.
    Returns int16 array:
        -1 = nodata, 0 = non-GDE, 1 = episodic, 2 = seasonal, 3 = perennial
    """
    if not Path(mask_tif).is_file():
        raise FileNotFoundError(f"Persistence mask not found: {mask_tif}")

    dst = np.full((len(tgt_lat), len(tgt_lon)), -1, dtype=np.int16)

    with rasterio.open(mask_tif) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=_dst_transform(tgt_lat, tgt_lon),
            dst_crs="EPSG:4326",
            dst_nodata=-1,
            resampling=Resampling.nearest,
        )

    return dst


# ── Cell area loading ────────────────────────────────────────────────────────────
def load_cell_area(tgt_lat: np.ndarray, tgt_lon: np.ndarray) -> np.ndarray:
    """
    Load cell area from NetCDF, align it to the target grid, and return a 2D array.

    Output units are km². If source units are m², values are converted.
    """
    if not Path(CELL_AREA_FILE).is_file():
        raise FileNotFoundError(f"Cell-area file not found: {CELL_AREA_FILE}")

    ds = _std(xr.open_dataset(CELL_AREA_FILE, decode_times=False, mask_and_scale=False))

    if CELL_AREA_VAR not in ds.data_vars:
        available = ", ".join(ds.data_vars)
        ds.close()
        raise KeyError(
            f"Variable '{CELL_AREA_VAR}' not found in {CELL_AREA_FILE}. "
            f"Available variables: {available}"
        )

    da = ds[CELL_AREA_VAR]

    if "time" in da.dims:
        da = da.isel(time=0, drop=True)

    if "lat" not in da.dims or "lon" not in da.dims:
        ds.close()
        raise ValueError(
            f"Cell-area variable '{CELL_AREA_VAR}' must have lat/lon dimensions."
        )

    grid_mismatch = (
        da.sizes.get("lat") != len(tgt_lat)
        or da.sizes.get("lon") != len(tgt_lon)
        or not np.array_equal(da["lat"].values, tgt_lat)
        or not np.array_equal(da["lon"].values, tgt_lon)
    )

    if grid_mismatch:
        da = da.interp(lat=tgt_lat, lon=tgt_lon, method="nearest")

    cell_area = da.values.astype(np.float64)
    ds.close()

    units = CELL_AREA_UNITS.lower()
    if units == "m2":
        cell_area = cell_area / 1e6
    elif units == "km2":
        pass
    else:
        raise ValueError(
            f"Unsupported CELL_AREA_UNITS='{CELL_AREA_UNITS}'. Use 'm2' or 'km2'."
        )

    return cell_area


# ── Area summation ───────────────────────────────────────────────────────────────
def compute_area(mask: np.ndarray, wet_vals: np.ndarray, cell_area: np.ndarray) -> float:
    """Return total wetGDE area in km² within mask."""
    return float((wet_vals[mask] * cell_area[mask]).sum())