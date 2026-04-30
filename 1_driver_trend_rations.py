

from __future__ import annotations
import gc
import os
import sys
import time
import logging
from pathlib import Path

import numpy as np
import dask
import dask.dataframe as dd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.cm import ScalarMappable
import cartopy.crs as ccrs

sys.path.append("/home/otoo0001/github/paper_3/future_gdes_new_v2026/wetgde_model")
from qa_utils import build_qa_mask

NCORES = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
dask.config.set(scheduler="threads", num_workers=NCORES)
dask.config.set({"array.slicing.split_large_chunks": True})
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
log = logging.getLogger("dom_driver_plus")

# ── paths ──────────────────────────────────────────────────────────────
PARQUET       = Path("/gpfs/scratch1/shared/otoo0001/data/data_June/regression_tiles_optimized/regression_results.parquet")
QA_DIR        = "/scratch-shared/otoo0001/from_projects/futurewetgde_todelete/quality_flags/"
THEILSEN_ZARR = "/projects/prjs1222/chapter2/output/9_analyse_historical/thiel_sen/wtd_trend_l2.zarr"
OUT_DIR       = Path("/scratch-shared/otoo0001/paper_2/revisions/figures/attributions/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── style ──────────────────────────────────────────────────────────────
PVALUE_THRESH = 0.05
DPI           = 1200
VMIN, VMAX    = -0.2, 0.2
NORM          = Normalize(VMIN, VMAX)
CMAP_MAIN     = "seismic_r"

MASK_COLOR = "#f8e0bd"
NSIG_COLOR = "#d9d9d9"

DRIVER_COLORS = {
    0: np.array([0.17, 0.33, 0.73], dtype=np.float32),   # recharge
    1: np.array([0.84, 0.10, 0.11], dtype=np.float32),   # abstraction
    2: np.array([0.10, 0.55, 0.25], dtype=np.float32),   # interaction
}
DRIVER_LABELS = {0: "Recharge", 1: "Abstraction", 2: "Interaction"}

RATIO_COLS = ["ratio_r", "ratio_q", "ratio_rq"]
BETA_COLS  = ["beta1", "beta2", "beta3"]
PVAL_COLS  = ["pval_b1", "pval_b2", "pval_b3"]

COEF_LABELS = {
    "beta1": r"$\beta_{\mathrm{recharge}}$",
    "beta2": r"$\beta_{\mathrm{abstraction}}$",
    "beta3": r"$\beta_{\mathrm{interaction}}$",
}

# ── performance ────────────────────────────────────────────────────────
CHUNKS = {"latitude": 1024, "longitude": 1440}

TERNARY_REQUIRE_SIG = True
TERNARY_ALPHA       = 0.35
TERNARY_SIZE        = 0.12

# ── Theil-Sen inset config ─────────────────────────────────────────────
TS_REGIONS = [
    ("(b) Western US",     (-125, -105, 30, 50)),
    ("(c) Indo-Gangetic",  (68, 90, 20, 33)),
    ("(d) North China",    (112, 124, 30, 42)),
    ("(e) Murray-Darling", (138, 153, -39, -26)),
]

# ── font sizes ─────────────────────────────────────────────────────────
FS_BASE      = 13
FS_TITLE     = 20
FS_PANEL_LBL = 24
FS_LEGEND    = 20
FS_TERNARY   = 20
FS_INSET     = 15

plt.rcParams.update({
    "font.size":       FS_BASE,
    "axes.titlesize":  FS_TITLE,
    "axes.labelsize":  FS_TITLE,
    "xtick.labelsize": FS_BASE,
    "ytick.labelsize": FS_BASE,
    "legend.fontsize": FS_LEGEND,
})


# ── helpers ────────────────────────────────────────────────────────────
def tic():
    return time.perf_counter()

def _nn_idx(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(src, tgt)
    idx = np.clip(idx, 1, len(src) - 1)
    left  = src[idx - 1]
    right = src[idx]
    return idx - ((tgt - left) <= (right - tgt)).astype(np.intp)

def to_180(lon_da):
    return ((lon_da + 180) % 360) - 180

def normalize_lon(ds, lon_name="longitude"):
    if float(ds[lon_name].max()) > 180.0:
        ds = ds.assign_coords({lon_name: to_180(ds[lon_name])}).sortby(lon_name)
    return ds

def _edges(coord_1d: np.ndarray) -> np.ndarray:
    d = float(np.diff(coord_1d)[0])
    return np.concatenate([coord_1d - d / 2, [coord_1d[-1] + d / 2]])

def _panel_label(ax, txt, fontsize=None):
    if fontsize is None:
        fontsize = FS_PANEL_LBL
    ax.text(
        0.01, 0.97, txt, transform=ax.transAxes,
        ha="left", va="top", fontsize=fontsize, fontweight="bold",
        color="black", zorder=999,
        path_effects=[pe.Stroke(linewidth=2.5, foreground="white"), pe.Normal()]
    )

def _base_map(ax, resolution="110m"):
    ax.set_facecolor("white")
    ax.coastlines(resolution=resolution, color="k", lw=0.3)
    ax.axis("off")

def add_colorbar(fig, cax, label, fontsize=None):
    if fontsize is None:
        fontsize = FS_TITLE
    cb = fig.colorbar(
        ScalarMappable(norm=NORM, cmap=CMAP_MAIN),
        cax=cax, orientation="vertical"
    )
    cb.set_label(label, fontsize=fontsize, labelpad=10)
    cb.outline.set_visible(False)
    for sp in cb.ax.spines.values():
        sp.set_visible(False)
    cb.set_ticks([-0.20, -0.10, 0.0, 0.10, 0.20])
    cb.ax.tick_params(size=0, labelsize=FS_BASE)
    return cb

def slice_box_da(da, lon_min, lon_max, lat_min, lat_max,
                 lon_name="longitude", lat_name="latitude"):
    lon = da[lon_name].values
    lat = da[lat_name].values
    lon_sl = slice(lon_min, lon_max) if lon[0] < lon[-1] else slice(lon_max, lon_min)
    lat_sl = slice(lat_min, lat_max) if lat[0] < lat[-1] else slice(lat_max, lat_min)
    return da.sel({lon_name: lon_sl, lat_name: lat_sl})

def _add_box(ax, extent, lw=2.8, color="k"):
    lon_min, lon_max, lat_min, lat_max = extent
    xs = [lon_min, lon_max, lon_max, lon_min, lon_min]
    ys = [lat_min, lat_min, lat_max, lat_max, lat_min]
    ax.plot(
        xs, ys, transform=ccrs.PlateCarree(),
        color=color, lw=lw, zorder=1000, solid_joinstyle="miter",
        path_effects=[pe.Stroke(linewidth=lw + 2.0, foreground="white"), pe.Normal()],
        clip_on=False
    )


def _pcolormesh_map(ax, lon_vals, lat_vals, data_2d,
                    cmap=CMAP_MAIN, norm=NORM, zorder=4):
    """
    Plot a 2-D array with pcolormesh on a cartopy axis.
    NaN cells are transparent (not plotted).
    """
    lon_e = _edges(lon_vals)
    lat_e = _edges(lat_vals)
    return ax.pcolormesh(
        lon_e, lat_e, data_2d,
        cmap=cmap, norm=norm,
        transform=ccrs.PlateCarree(),
        shading="auto", zorder=zorder,
        rasterized=True
    )


def _pcolormesh_solid(ax, lon_vals, lat_vals, mask_2d, color, zorder=3):
    """
    Plot a solid-colour pcolormesh for boolean mask regions
    (e.g. insignificant or QA-excluded cells).
    """
    if not mask_2d.any():
        return
    cmap_solid = ListedColormap([color])
    data = np.where(mask_2d, 1.0, np.nan).astype(np.float32)
    lon_e = _edges(lon_vals)
    lat_e = _edges(lat_vals)
    ax.pcolormesh(
        lon_e, lat_e, data,
        cmap=cmap_solid, vmin=0, vmax=1,
        transform=ccrs.PlateCarree(),
        shading="auto", zorder=zorder,
        rasterized=True
    )


# ── ternary ────────────────────────────────────────────────────────────
def _tern_to_xy(r, q, rq):
    h = np.sqrt(3) / 2.0
    return r + 0.5 * q, h * q


def draw_ternary_pixels(ax, ratios, colors, alpha=0.35, size=0.12):
    print("[draw_ternary_pixels] Start...", flush=True)
    t = tic()

    r  = ratios[:, 0]
    q  = ratios[:, 1]
    rq = ratios[:, 2]
    h  = np.sqrt(3) / 2.0
    x, y = _tern_to_xy(r, q, rq)

    ax.scatter(x, y, c=colors, s=size, linewidths=0, alpha=alpha, rasterized=True)
    ax.plot([0, 1, 0.5, 0], [0, 0, h, 0], color="black", lw=1.2, zorder=5)

    gl_kw = dict(color="#cccccc", lw=0.4, zorder=0, linestyle="--")
    for f in [0.25, 0.50, 0.75]:
        x0, y0 = _tern_to_xy(1-f, 0,   f);   x1, y1 = _tern_to_xy(0,   1-f, f)
        ax.plot([x0, x1], [y0, y1], **gl_kw)
        x0, y0 = _tern_to_xy(f,   0,   1-f); x1, y1 = _tern_to_xy(f,   1-f, 0)
        ax.plot([x0, x1], [y0, y1], **gl_kw)
        x0, y0 = _tern_to_xy(1-f, f,   0);   x1, y1 = _tern_to_xy(0,   f,   1-f)
        ax.plot([x0, x1], [y0, y1], **gl_kw)

    def perp(v): return np.array([-v[1], v[0]])
    norm_rc  = np.array([-0.5, h]);  norm_rc  /= np.linalg.norm(norm_rc)
    norm_lc  = np.array([-0.5, -h]); norm_lc /= np.linalg.norm(norm_lc)
    norm_r   = np.array([1.0, 0.0])
    tick_len = 0.022
    p_bot = perp(norm_r); p_re = perp(norm_rc); p_le = perp(norm_lc)

    for f in [0.25, 0.50, 0.75]:
        px, py = 1.0 - f, 0.0
        ax.plot([px - tick_len*p_bot[0], px + tick_len*p_bot[0]],
                [py - tick_len*p_bot[1], py + tick_len*p_bot[1]], color="black", lw=0.8, zorder=6)
        px, py = 1.0 + f*(-0.5), f*h
        ax.plot([px - tick_len*p_re[0], px + tick_len*p_re[0]],
                [py - tick_len*p_re[1], py + tick_len*p_re[1]], color="black", lw=0.8, zorder=6)
        px, py = 0.5 + f*(-0.5), h + f*(-h)
        ax.plot([px - tick_len*p_le[0], px + tick_len*p_le[0]],
                [py - tick_len*p_le[1], py + tick_len*p_le[1]], color="black", lw=0.8, zorder=6)

    fs_lbl = FS_TERNARY + 1
    ax.text(0.50,  h + 0.08, "Human impact",   ha="center", va="bottom",
            fontsize=fs_lbl, fontweight="bold", color=DRIVER_COLORS[1], zorder=10)
    ax.text(-0.03, -0.07, "Combined impact",    ha="center", va="top",
            fontsize=fs_lbl, fontweight="bold", color=DRIVER_COLORS[2], zorder=10)
    ax.text(1.03,  -0.07, "Climate impact",     ha="center", va="top",
            fontsize=fs_lbl, fontweight="bold", color=DRIVER_COLORS[0], zorder=10)

    ax.set_xlim(-0.15, 1.15); ax.set_ylim(-0.15, h + 0.18)
    ax.set_aspect("equal"); ax.axis("off")
    print(f"  Pixels plotted: {len(ratios):,}  ({tic()-t:.1f}s)\n", flush=True)


# ── parquet rasterization ──────────────────────────────────────────────
def rasterize_parquet(df, qa_2d, lat_idx, lon_idx, lat_vals, lon_vals):
    """
    Rasterize parquet point data onto the parquet lat/lon grid.

    Returns
    -------
    driver_rgba  : (nlat, nlon, 4) uint8   RGBA for dominant driver map
    beta_grids   : dict key -> (nlat, nlon) float32  NaN where not sig
    nsig_2d      : (nlat, nlon) bool   QA-valid but not significant
    excl_2d      : (nlat, nlon) bool   QA excluded
    ternary_out  : (ratios, colors, n_valid, dom_counts)
    """
    print("\n[rasterize_parquet] Start...", flush=True)
    t = tic()

    N      = len(df)
    nlat   = len(lat_vals)
    nlon   = len(lon_vals)
    lons   = df["lon"].values.astype(np.float32)
    lats   = df["lat"].values.astype(np.float32)
    betas  = np.abs(df[BETA_COLS].values.astype(np.float32))
    raw_b  = df[BETA_COLS].values.astype(np.float32)
    pvals  = df[PVAL_COLS].values.astype(np.float32)
    ratios = df[RATIO_COLS].values.astype(np.float32)

    qa_row   = qa_2d[lat_idx, lon_idx]
    dom_idx  = np.argmax(betas, axis=1)
    dom_pval = pvals[np.arange(N), dom_idx]

    sig   = (dom_pval <= PVALUE_THRESH) & qa_row
    insig = (~(dom_pval <= PVALUE_THRESH)) & qa_row
    excl  = ~qa_row

    print(f"  sig={sig.sum():,}  insig={insig.sum():,}  excl={excl.sum():,}", flush=True)

    # ── 2-D boolean masks ──────────────────────────────────────────────
    nsig_2d = np.zeros((nlat, nlon), dtype=bool)
    excl_2d = np.zeros((nlat, nlon), dtype=bool)
    nsig_2d[lat_idx[insig], lon_idx[insig]] = True
    excl_2d[lat_idx[excl],  lon_idx[excl]]  = True

    # ── dominant driver RGBA grid ──────────────────────────────────────
    # Encode driver as integer: 0/1/2 for sig cells, 3=insig, 4=excl, 255=empty
    driver_grid = np.full((nlat, nlon), 255, dtype=np.uint8)
    driver_grid[lat_idx[excl],  lon_idx[excl]]  = 4
    driver_grid[lat_idx[insig], lon_idx[insig]] = 3
    for k in DRIVER_COLORS:
        m = sig & (dom_idx == k)
        driver_grid[lat_idx[m], lon_idx[m]] = k
        print(f"    {DRIVER_LABELS[k]:15s}: {m.sum():,}", flush=True)

    # ── beta grids ────────────────────────────────────────────────────
    beta_grids = {}
    for j, key in enumerate(BETA_COLS):
        sig_j = (pvals[:, j] <= PVALUE_THRESH) & qa_row
        arr   = np.full((nlat, nlon), np.nan, dtype=np.float32)
        arr[lat_idx[sig_j], lon_idx[sig_j]] = raw_b[sig_j, j]
        beta_grids[key] = arr
        print(f"  {key}: sig={sig_j.sum():,}", flush=True)

    # ── ternary ───────────────────────────────────────────────────────
    keep = sig
    rr   = ratios[keep, 0]; qq = ratios[keep, 1]; rrq = ratios[keep, 2]
    dom  = dom_idx[keep]
    s    = rr + qq + rrq
    good = np.isfinite(s) & (s > 0)
    rr   = rr[good] / s[good]; qq = qq[good] / s[good]; rrq = rrq[good] / s[good]
    dom  = dom[good]

    tern_colors = np.zeros((len(dom), 3), dtype=np.float32)
    for k, col in DRIVER_COLORS.items():
        tern_colors[dom == k] = col

    dom_counts  = {k: int((dom == k).sum()) for k in DRIVER_LABELS}
    n_valid     = int(good.sum())
    ternary_out = (np.column_stack([rr, qq, rrq]).astype(np.float32),
                   tern_colors, n_valid, dom_counts)

    del betas, raw_b, pvals, ratios, dom_idx, dom_pval, sig, insig, excl, qa_row
    gc.collect()

    print(f"[rasterize_parquet] Done in {tic()-t:.1f}s\n", flush=True)
    return driver_grid, beta_grids, nsig_2d, excl_2d, ternary_out


# ── Theil-Sen load ─────────────────────────────────────────────────────
def load_theilsen_arrays(qa_2d_ts, lat_vals_ts, lon_vals_ts):
    """
    Single full zarr read. Negate slope: declining heads -> red.

    Returns
    -------
    slope_sig : (nlat, nlon) float32  negated slope, NaN where not sig
    nsig_2d   : (nlat, nlon) bool
    excl_2d   : (nlat, nlon) bool
    """
    t = tic()
    nlat, nlon = len(lat_vals_ts), len(lon_vals_ts)
    print(f"\n[load_theilsen_arrays] {nlat}×{nlon}...", flush=True)

    ds = normalize_lon(xr.open_zarr(THEILSEN_ZARR, chunks=CHUNKS))
    print("  Reading slope...", flush=True)
    sl = ds["slope"].values.astype(np.float32)
    print("  Reading pval...", flush=True)
    pv = ds["p"].values.astype(np.float32)
    print(f"  Read in {tic()-t:.1f}s", flush=True)

    fin  = np.isfinite(sl)
    sig  = (pv <= PVALUE_THRESH) & qa_2d_ts & fin
    nsig = fin & qa_2d_ts & ~sig
    excl = fin & ~qa_2d_ts

    print(f"  sig={sig.sum():,}  nsig={nsig.sum():,}  excl={excl.sum():,}", flush=True)

    slope_sig = np.full((nlat, nlon), np.nan, dtype=np.float32)
    slope_sig[sig] = -sl[sig]   # negate: declining heads -> negative -> red

    del sl, pv, fin
    gc.collect()

    print(f"[load_theilsen_arrays] Done in {tic()-t:.1f}s\n", flush=True)
    return slope_sig, nsig.astype(bool), excl.astype(bool)


# ── figures ────────────────────────────────────────────────────────────
def make_dominant_driver_figure(driver_grid, nsig_2d, excl_2d,
                                lat_vals, lon_vals,
                                ternary_ratios, ternary_colors,
                                n_pixels, dom_counts):
    """
    Global dominant driver map using pcolormesh per driver category,
    same rendering style as insets.
    """
    print("[figure] Dominant driver...", flush=True)
    t = tic()

    lon_e = _edges(lon_vals)
    lat_e = _edges(lat_vals)

    fig    = plt.figure(figsize=(20, 11), facecolor="white")
    ax_map = fig.add_axes([0.01, 0.18, 0.70, 0.72], projection=ccrs.Robinson())
    ax_tern = fig.add_axes([0.74, 0.34, 0.23, 0.44])

    ax_map.set_global()
    _base_map(ax_map)

    # Draw in z-order: excl (background) -> insig (grey) -> each driver
    _pcolormesh_solid(ax_map, lon_vals, lat_vals, excl_2d,  MASK_COLOR, zorder=2)
    _pcolormesh_solid(ax_map, lon_vals, lat_vals, nsig_2d,  NSIG_COLOR, zorder=3)

    for k, col in DRIVER_COLORS.items():
        mask = (driver_grid == k)
        if mask.any():
            hex_col = matplotlib.colors.to_hex(col)
            _pcolormesh_solid(ax_map, lon_vals, lat_vals, mask, hex_col, zorder=4 + k)

    draw_ternary_pixels(ax_tern, ternary_ratios, ternary_colors,
                        alpha=TERNARY_ALPHA, size=TERNARY_SIZE)
    _panel_label(ax_tern, "(b)")

    out = OUT_DIR / "dominant_driver_ratio_ternary.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved -> {out}  ({tic()-t:.1f}s)", flush=True)


def make_coefficients_figure(beta_grids, nsig_2d, excl_2d, lat_vals, lon_vals):
    """3 rows × 1 col, pcolormesh per panel."""
    print("[figure] OLS coefficients (3-row layout)...", flush=True)
    t0 = tic()

    fig   = plt.figure(figsize=(12, 14), facecolor="white")
    outer = gridspec.GridSpec(
        1, 2, width_ratios=[1, 0.035], wspace=0.06,
        left=0.02, right=0.95, top=0.97, bottom=0.03)
    inner = gridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=outer[0, 0], hspace=0.08)

    for j, key in enumerate(BETA_COLS):
        ax = fig.add_subplot(inner[j, 0], projection=ccrs.Robinson())
        ax.set_global()
        _base_map(ax)

        # excl and insig solid layers first
        _pcolormesh_solid(ax, lon_vals, lat_vals, excl_2d, MASK_COLOR, zorder=2)
        _pcolormesh_solid(ax, lon_vals, lat_vals, nsig_2d, NSIG_COLOR, zorder=3)

        # significant values through seismic_r colormap
        _pcolormesh_map(ax, lon_vals, lat_vals, beta_grids[key],
                        cmap=CMAP_MAIN, norm=NORM, zorder=4)

        ax.set_title(COEF_LABELS[key], fontsize=FS_TITLE + 2, pad=8)
        _panel_label(ax, ["(a)", "(b)", "(c)"][j])

    cax = fig.add_axes([0.96, 0.25, 0.015, 0.50])
    add_colorbar(fig, cax, r"Coefficient", fontsize=FS_TITLE)

    out = OUT_DIR / "ols_coefficients_3panel.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved -> {out}  ({tic()-t0:.1f}s)", flush=True)


def make_theilsen_figure(slope_sig, nsig_2d, excl_2d, lat_vals_ts, lon_vals_ts):
    """
    Global Theil-Sen map + four inset panels, all using pcolormesh.
    slope_sig is already negated: declining heads -> red, rising -> blue.
    """
    print("[figure] Theil-Sen with insets...", flush=True)
    t0 = tic()

    slope_sig_da = xr.DataArray(
        slope_sig,
        coords={"latitude": lat_vals_ts, "longitude": lon_vals_ts},
        dims=("latitude", "longitude"),
        name="slope_sig"
    )

    fig = plt.figure(figsize=(18, 10.5), facecolor="white")
    gs  = gridspec.GridSpec(
        nrows=2, ncols=5,
        height_ratios=[3.3, 1.45],
        width_ratios=[1, 1, 1, 1, 0.05],
        hspace=0.06, wspace=0.04,
        left=0.02, right=0.96, top=0.93, bottom=0.05
    )

    # Global map
    ax = fig.add_subplot(gs[0, 0:4], projection=ccrs.Robinson())
    ax.set_global()
    _base_map(ax)

    _pcolormesh_solid(ax, lon_vals_ts, lat_vals_ts, excl_2d, MASK_COLOR, zorder=2)
    _pcolormesh_solid(ax, lon_vals_ts, lat_vals_ts, nsig_2d, NSIG_COLOR, zorder=3)
    _pcolormesh_map(ax, lon_vals_ts, lat_vals_ts, slope_sig,
                    cmap=CMAP_MAIN, norm=NORM, zorder=4)

    _panel_label(ax, "(a)")

    # Inset panels — identical pcolormesh approach
    for j, (lab, ext) in enumerate(TS_REGIONS):
        print(f"  Inset {j+1}/4: {lab}", flush=True)
        sub = slice_box_da(slope_sig_da, *ext)
        sub = sub.compute() if hasattr(sub.data, "compute") else sub

        lon_e = _edges(sub.longitude.values)
        lat_e = _edges(sub.latitude.values)

        ax_in = fig.add_subplot(gs[1, j], projection=ccrs.PlateCarree())
        ax_in.set_extent(ext, crs=ccrs.PlateCarree())
        ax_in.set_facecolor("white")
        ax_in.coastlines(resolution="50m", color="k", lw=0.4)
        ax_in.axis("off")
        ax_in.pcolormesh(
            lon_e, lat_e, sub.values,
            cmap=CMAP_MAIN, norm=NORM,
            transform=ccrs.PlateCarree(),
            shading="auto", zorder=4, rasterized=True
        )
        ax_in.text(0.5, -0.06, lab, transform=ax_in.transAxes,
                   ha="center", va="top", fontsize=FS_INSET)

    # Region boxes on global map
    for _, ext in TS_REGIONS:
        _add_box(ax, ext, lw=2.8, color="k")

    # Colorbar
    cax = fig.add_subplot(gs[0, 4])
    add_colorbar(fig, cax, r"Slope (m yr$^{-1}$)", fontsize=FS_TITLE)

    del slope_sig_da
    gc.collect()

    out = OUT_DIR / "theilsen_slope_map_with_insets.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved -> {out}  ({tic()-t0:.1f}s)", flush=True)


# ── main ───────────────────────────────────────────────────────────────
def main():
    t0 = tic()
    print("=" * 70, flush=True)
    print("plot_dominant_driver_ratio_with_theilsen_and_coefs.py", flush=True)
    print(f"SLURM cores  : {NCORES}", flush=True)
    print(f"OUT_DIR      : {OUT_DIR}", flush=True)
    print(f"Chunks       : {CHUNKS}", flush=True)
    print("=" * 70, flush=True)

    # [1/5] QA mask
    print(f"\n[1/5] Loading QA mask...", flush=True)
    t = tic()
    qa_lat_g, qa_lon_g, qa_mask_global = build_qa_mask(QA_DIR)
    if qa_lat_g[0] > qa_lat_g[-1]:
        qa_lat_g       = qa_lat_g[::-1]
        qa_mask_global = qa_mask_global[::-1, :]
    print(f"  QA grid: {qa_mask_global.shape}  valid={qa_mask_global.mean()*100:.1f}%"
          f"  ({tic()-t:.1f}s)", flush=True)

    # [2/5] Parquet
    print(f"\n[2/5] Loading parquet...", flush=True)
    t    = tic()
    cols = ["lat", "lon"] + BETA_COLS + PVAL_COLS + RATIO_COLS
    df   = dd.read_parquet(PARQUET, columns=cols,
                           engine="pyarrow").dropna(subset=cols).compute()
    print(f"  Rows={len(df):,}  mem={df.memory_usage(deep=True).sum()/1e9:.2f}GB"
          f"  ({tic()-t:.1f}s)", flush=True)

    lat_vals = np.sort(df["lat"].unique())
    lon_vals = np.sort(df["lon"].unique())
    lat_idx  = np.searchsorted(lat_vals, df["lat"].values)
    lon_idx  = np.searchsorted(lon_vals, df["lon"].values)

    # [3/5] QA reindex
    print(f"\n[3/5] QA reindex...", flush=True)
    t  = tic()
    li = _nn_idx(qa_lat_g, lat_vals)
    lj = _nn_idx(qa_lon_g, lon_vals)
    qa_2d = qa_mask_global[np.ix_(li, lj)]
    print(f"  QA valid: {qa_2d.mean()*100:.1f}%  ({tic()-t:.1f}s)", flush=True)

    # [4/5] Rasterize parquet
    print(f"\n[4/5] Rasterizing parquet data...", flush=True)
    driver_grid, beta_grids, nsig_2d, excl_2d, ternary_out = rasterize_parquet(
        df, qa_2d, lat_idx, lon_idx, lat_vals, lon_vals)
    ternary_ratios, ternary_colors, n_ternary_pixels, dom_counts = ternary_out

    del df, qa_2d, li, lj
    gc.collect()

    # [5/5] Theil-Sen
    print(f"\n[5/5] Loading Theil-Sen zarr...", flush=True)
    ds_ts       = normalize_lon(xr.open_zarr(THEILSEN_ZARR, chunks=CHUNKS))
    lat_vals_ts = ds_ts["latitude"].values
    lon_vals_ts = ds_ts["longitude"].values
    li_ts       = _nn_idx(qa_lat_g, lat_vals_ts)
    lj_ts       = _nn_idx(qa_lon_g, lon_vals_ts)
    qa_2d_ts    = qa_mask_global[np.ix_(li_ts, lj_ts)]
    del qa_mask_global
    gc.collect()

    slope_sig, ts_nsig_2d, ts_excl_2d = load_theilsen_arrays(
        qa_2d_ts, lat_vals_ts, lon_vals_ts)
    del qa_2d_ts
    gc.collect()

    # Render
    print("\n[plots] Rendering...", flush=True)

    make_dominant_driver_figure(
        driver_grid, nsig_2d, excl_2d,
        lat_vals, lon_vals,
        ternary_ratios, ternary_colors,
        n_pixels=n_ternary_pixels,
        dom_counts=dom_counts)
    del driver_grid, ternary_ratios, ternary_colors
    gc.collect()

    make_coefficients_figure(beta_grids, nsig_2d, excl_2d, lat_vals, lon_vals)
    del beta_grids, nsig_2d, excl_2d
    gc.collect()

    make_theilsen_figure(slope_sig, ts_nsig_2d, ts_excl_2d, lat_vals_ts, lon_vals_ts)
    del slope_sig, ts_nsig_2d, ts_excl_2d
    gc.collect()

    print(f"\n{'='*70}", flush=True)
    print(f"All done in {tic()-t0:.1f}s  |  outputs in {OUT_DIR}", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
    