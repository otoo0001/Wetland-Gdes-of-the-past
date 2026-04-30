#!/bin/env python3
"""
Average number of groundwater dependency decadal maps — full resolution, 900 dpi, datashader rendering

Inputs (pre-computed with CDO):
  avg_wet_1980s.nc
  avg_wet_2010s.nc
  diff_2010s_minus_1980s.nc

Outputs (to OUT_DIR):
  1) main_2010s_wet_months.pdf/.png
  2) SI_1980s_and_difference.pdf/.png
"""

from pathlib import Path
import time
import numpy as np
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.colors import PowerNorm, TwoSlopeNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import datashader as ds

# ── I/O ───────────────────────────────────────────────────────────
DATA_DIR     = Path("/gpfs/scratch1/shared/otoo0001/past_gde")
OUT_DIR      = Path("/scratch-shared/otoo0001/paper_2/revisions/figures/decadal_maps/")
PERSIST_MASK = Path(
    "/scratch-shared/otoo0001/paper_2/revisions/shapefiles_from/"
    "wetgde_persistence_mask_historical.tif"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

F_1980s = DATA_DIR / "avg_wet_1980s.nc"
F_2010s = DATA_DIR / "avg_wet_2010s.nc"
F_DIFF  = DATA_DIR / "diff_2010s_minus_1980s.nc"

# ── Canvas resolution @ 900 dpi ───────────────────────────────────
CANVAS_W,    CANVAS_H    = 12600, 7200   # main:  14 x 8 in
CANVAS_W_SI, CANVAS_H_SI = 12600, 6300  # SI:    14 x 7 in per panel

# ── Timer ─────────────────────────────────────────────────────────
class Timer:
    def __init__(self):
        self._t0 = self._step = time.perf_counter()

    def tick(self, label=""):
        now = time.perf_counter()
        print(
            f"[{label:<45s}]  step={now-self._step:6.1f}s  total={now-self._t0:6.1f}s",
            flush=True,
        )
        self._step = now

T = Timer()

# ── Colormaps ─────────────────────────────────────────────────────
SEQ_CMAP = matplotlib.colormaps["YlGnBu"].copy()
SEQ_CMAP.set_bad(color="white")
SEQ_CMAP.set_under(color="white")

DIV_CMAP = matplotlib.colormaps["seismic_r"].copy()
DIV_CMAP.set_bad(color="white")

# ── Load helpers ──────────────────────────────────────────────────
SKIP_VARS = {"time_bnds", "lat_bnds", "lon_bnds", "latitude_bnds", "longitude_bnds"}

def load_var(path: Path) -> xr.DataArray:
    ds_obj = xr.open_dataset(path, decode_times=False)
    var    = next(v for v in ds_obj.data_vars if v not in SKIP_VARS)
    da     = ds_obj[var].squeeze()
    print(f"[LOAD] {path.name}  var='{var}'  shape={dict(da.sizes)}  dtype={da.dtype}", flush=True)
    print(f"       coords: {list(da.coords)}", flush=True)
    return da

# ── Persistence mask ──────────────────────────────────────────────
def load_persistence_mask(tif_path: Path, tgt_lat: np.ndarray, tgt_lon: np.ndarray) -> np.ndarray:
    if not tif_path.exists():
        print(f"[WARN] Persistence mask not found: {tif_path}, no GDE masking applied", flush=True)
        return np.ones((len(tgt_lat), len(tgt_lon)), dtype=bool)

    try:
        import rasterio
        with rasterio.open(str(tif_path)) as src:
            data     = src.read(1)
            nodata_v = src.nodata
            src_h, src_w = data.shape
            src_lat  = np.linspace(src.bounds.top,  src.bounds.bottom, src_h)
            src_lon  = np.linspace(src.bounds.left, src.bounds.right,  src_w)

        lat_idx   = np.clip(np.searchsorted(-src_lat, -tgt_lat), 0, src_h - 1)
        lon_idx   = np.clip(np.searchsorted(src_lon,   tgt_lon), 0, src_w - 1)
        resampled = data[np.ix_(lat_idx, lon_idx)]

        if nodata_v is not None:
            resampled = np.where(resampled == nodata_v, 0, resampled)

        gde_mask = resampled > 0
        print(f"[MASK] GDE cells: {gde_mask.sum():,}", flush=True)
        return gde_mask

    except Exception as e:
        print(f"[WARN] Could not load persistence mask ({e}), no GDE masking applied", flush=True)
        return np.ones((len(tgt_lat), len(tgt_lon)), dtype=bool)

# ── Coordinate helpers ────────────────────────────────────────────
def _get_lonlat(da: xr.DataArray):
    lon_names = {"lon", "longitude", "x"}
    lat_names = {"lat", "latitude", "y"}
    dims      = set(da.dims) | set(da.coords)
    lon_dim   = next(d for d in dims if d.lower() in lon_names)
    lat_dim   = next(d for d in dims if d.lower() in lat_names)
    return da[lon_dim].values.astype("float64"), da[lat_dim].values.astype("float64")

def _lon_lat_extent(da: xr.DataArray):
    lons, lats = _get_lonlat(da)
    return [float(lons.min()), float(lons.max())], \
           [float(lats.min()), float(lats.max())]

# ── Datashader rasterizer ─────────────────────────────────────────
def rasterize(data2d: xr.DataArray, norm, cmap,
              canvas_w: int, canvas_h: int) -> np.ndarray:
    lons, lats = _get_lonlat(data2d)

    # clip Antarctica
    lat_mask = lats >= -60.0
    lats     = lats[lat_mask]
    vals     = data2d.values.astype("float64")[lat_mask, :]

    da2 = xr.DataArray(
        vals, dims=["y", "x"],
        coords={"y": lats, "x": lons},
    )
    cvs = ds.Canvas(
        plot_width  = canvas_w,
        plot_height = canvas_h,
        x_range     = (float(lons.min()), float(lons.max())),
        y_range     = (float(lats.min()), float(lats.max())),
    )
    agg      = cvs.raster(da2, interpolate="nearest")
    agg_vals = np.array(agg)

    normed     = norm(agg_vals)
    rgba       = cmap(normed)
    mask       = np.ma.getmaskarray(normed) | ~np.isfinite(agg_vals)
    rgba[mask] = (1.0, 1.0, 1.0, 1.0)   # white for NaN / masked

    print(
        f"[RASTERIZE] canvas=({canvas_w}x{canvas_h})  "
        f"white={mask.sum():,}  mem={rgba.nbytes/1e6:.0f} MB",
        flush=True,
    )
    return (rgba * 255).astype(np.uint8)

# ── Map helpers ───────────────────────────────────────────────────
def _add_raster_to_ax(ax, rgba, lon_extent, lat_extent):
    ax.imshow(
        rgba,
        origin        = "upper",
        extent        = [lon_extent[0], lon_extent[1], lat_extent[0], lat_extent[1]],
        transform     = ccrs.PlateCarree(),
        interpolation = "nearest",
        aspect        = "auto",
        zorder        = 2,
    )

def _base_map(ax):
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="white", edgecolor="none", zorder=0)
    ax.axis("off")
    # paint white over Antarctica (Robinson ignores set_extent for clipping)
    ax.add_patch(mpatches.Rectangle(
        xy        = (-180, -90),
        width     = 360,
        height    = 30,
        transform = ccrs.PlateCarree(),
        facecolor = "white",
        edgecolor = "none",
        zorder    = 10,
    ))

def _colorbar(fig, ax_pos, norm, cmap, label, ticks, fontsize=11):
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cax = fig.add_axes(ax_pos)
    cb  = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=ticks)
    cb.set_label(label, fontsize=fontsize)
    cb.ax.tick_params(labelsize=fontsize)
    cb.outline.set_visible(False)
    return cb

def _save(fig, out_dir: Path, stem: str):
    for ext in ("pdf", "png"):
        out = out_dir / f"{stem}.{ext}"
        fig.savefig(out, dpi=900, bbox_inches="tight")
        print(f"[SAVE] {out}  ({out.stat().st_size/1e6:.1f} MB)", flush=True)

# ── Figures ───────────────────────────────────────────────────────
def plot_main_2010s(avg10: xr.DataArray, stem: str):
    norm = PowerNorm(0.5, vmin=1, vmax=12)
    data = avg10.where(avg10 > 0)
    rgba = rasterize(data, norm, SEQ_CMAP, CANVAS_W, CANVAS_H)
    T.tick("rasterize 2010s")

    lon_ext, lat_ext = _lon_lat_extent(data)
    lat_ext[0] = -60.0

    fig = plt.figure(figsize=(14, 8))
    ax  = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    _base_map(ax)
    _add_raster_to_ax(ax, rgba, lon_ext, lat_ext)
    _colorbar(fig, [0.25, 0.06, 0.5, 0.03], norm, SEQ_CMAP,
              "Average number of months per year", ticks=[1, 4, 7, 10, 12], fontsize=15)

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    _save(fig, OUT_DIR, stem)
    plt.close(fig)

def _colorbar(fig, ax_pos, norm, cmap, label, ticks, fontsize=11):
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cax = fig.add_axes(ax_pos)
    cb  = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=ticks)
    cb.set_label(label, fontsize=fontsize)
    cb.ax.tick_params(labelsize=fontsize)
    cb.outline.set_visible(False)
    return cb


def plot_SI_combined(avg80: xr.DataArray, diff: xr.DataArray,
                     gde_mask: np.ndarray, stem: str):
    norm_seq  = PowerNorm(0.5, vmin=1, vmax=12)
    norm_diff = TwoSlopeNorm(vcenter=0, vmin=-4, vmax=4)

    data80 = avg80.where(avg80 > 0)

    # apply GDE mask to diff: non-GDE land -> NaN -> white
    diff_vals            = diff.values.astype("float64")
    diff_vals[~gde_mask] = np.nan
    diff_masked          = diff.copy(data=diff_vals).where(
        xr.DataArray(gde_mask, dims=diff.dims, coords=diff.coords)
    )
    diff_plot = diff_masked.where(np.isfinite(diff_masked.values))

    rgba80    = rasterize(data80,    norm_seq,  SEQ_CMAP,  CANVAS_W_SI, CANVAS_H_SI)
    T.tick("rasterize 1980s")
    rgba_diff = rasterize(diff_plot, norm_diff, DIV_CMAP,  CANVAS_W_SI, CANVAS_H_SI)
    T.tick("rasterize diff")

    lon80,  lat80  = _lon_lat_extent(data80)
    lon_df, lat_df = _lon_lat_extent(diff_plot)
    lat80[0]  = -60.0
    lat_df[0] = -60.0

    fig = plt.figure(figsize=(14, 14))
    gs  = gridspec.GridSpec(2, 1, figure=fig, hspace=0.25)

    ax0 = fig.add_subplot(gs[0], projection=ccrs.Robinson())
    _base_map(ax0)
    ax0.text(0.01, 0.97, "(a)", transform=ax0.transAxes,
             fontsize=20, fontweight="bold", va="top", ha="left")
    _add_raster_to_ax(ax0, rgba80, lon80, lat80)
    
    ax0.set_title("(1980-1989)", fontsize=18)
    _colorbar(fig, [0.25, 0.52, 0.5, 0.02], norm_seq, SEQ_CMAP,
              "Average number of months per year", ticks=[1, 4, 7, 10, 12], fontsize=15)

    ax1 = fig.add_subplot(gs[1], projection=ccrs.Robinson())
    _base_map(ax1)
    ax1.text(0.01, 0.97, "(b)", transform=ax1.transAxes,
             fontsize=20, fontweight="bold", va="top", ha="left")
    _add_raster_to_ax(ax1, rgba_diff, lon_df, lat_df)
    ax1.set_title("(2010s - 1980s)", fontsize=18)
    _colorbar(fig, [0.25, 0.05, 0.5, 0.02], norm_diff, DIV_CMAP,
              "Difference in average number of months (2010s - 1980s)",
              ticks=[-4, -2, 0, 2, 4], fontsize=15)

    _save(fig, OUT_DIR, stem)
    plt.close(fig)

# ── Main ──────────────────────────────────────────────────────────
def main():
    T.tick("startup")

    avg80 = load_var(F_1980s)
    avg10 = load_var(F_2010s)
    diff  = load_var(F_DIFF)
    T.tick("load all three files")

    lons, lats = _get_lonlat(diff)
    gde_mask   = load_persistence_mask(PERSIST_MASK, lats, lons)
    T.tick("load persistence mask")

    plot_main_2010s(avg10, "main_2010s_wet_months")
    T.tick("plot + save main")

    plot_SI_combined(avg80, diff, gde_mask, "SI_1980s_and_difference")
    T.tick("plot + save SI")

    print(f"\n[DONE] all outputs -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()