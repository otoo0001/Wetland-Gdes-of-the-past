#!/usr/bin/env python3
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter, MaxNLocator
from matplotlib import cm, colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import theilslopes
from shapely.geometry import Polygon, MultiPolygon
import cartopy.crs as ccrs
import cartopy.feature as cfeature

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── paths ─────────────────────────────────────────────────────────────
INPUT_FOLDER      = "/archive/depfg/otoo0001/paper_2/output_gde"
AREA_FILE         = f"{INPUT_FOLDER}/gde_area_by_biome_1_14monthly.parquet"
AREA_REALM_FILE   = f"{INPUT_FOLDER}/gde_area_by_biome_realm_monthly.parquet"
ENSO_FILE         = f"{INPUT_FOLDER}/iersst_nino3_4_1979_2019.nc"
BIOME_SHP         = "/archive/depfg/otoo0001/data/shapefiles/biomes/biomes_new/biomes/wwf_terr_ecos.shp"
OUTPUT_FOLDER     = "/eejit/home/otoo0001/github/paper_2/wetland_gdes/time_series_area_newfor_pub"
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

ENSO_VAR          = "Nino3.4"
ENSO_START        = "1979-01-15"

# ── output options ────────────────────────────────────────────────────
DPI           = 900
COMPOSITE_DPI = 300
SAVE_PDF      = True
SAVE_SVG      = True

# ── plotting options ──────────────────────────────────────────────────
PLOT_AS_MKM2 = True
CLEAN_MODE   = "none"
CLEAN_SUBSET = [3, 7]
MAD_K        = 6.0

ONLY_TEXT    = "annual_pct_rate"
TEXT_CORNER  = "tr"
RANDOM_CORNER_DETERMINISTIC = True

PCT_RANGE_MODE     = "fixed"
PCT_VMIN, PCT_VMAX = -20, 20
REMOVE_ANTARCTICA  = True

MAKE_TRENDS       = True
MAKE_PCT_MAP      = True
MAKE_COMPOSITE    = True
MAKE_SUMMARIES    = True
MAKE_ENSO_SPATIAL = True   

# ENSO cross-correlation settings
MAX_LAG  = 12    
MIN_N    = 10    
CORR_MIN = 0.0   

# ── fonts — individual trend plots ────────────────────────────────────
FS_TREND_TITLE  = 68
FS_TREND_TICK   = 48
FS_TREND_ANN    = 54
FS_TREND_YLABEL = 48

# ── fonts — pct-change map ────────────────────────────────────────────
FS_PCT_MAP_TITLE  = 80
FS_PCT_MAP_CB     = 28
PCT_CB_WIDTH      = "1.8%"
PCT_CB_HEIGHT     = "50%"
PCT_CB_BBOX       = (-0.08, 0.25, 1, 1)

# ── fonts — global time-series panel (below pct map) ──────────────────
FS_PCT_MAP_TS_AX  = 20
FS_PCT_MAP_TS_LEG = 18

# ── fonts — ENSO spatial maps ─────────────────────────────────────────
FS_ENSO_MAP_TITLE = 40
FS_ENSO_MAP_CB    = 28

# ── fonts — composite panels ──────────────────────────────────────────
FS_COMP_PANEL_TITLE  = 64
FS_COMP_PANEL_TICK   = 38
FS_COMP_PANEL_ANN    = 46
FS_COMP_PANEL_YLABEL = 40
FS_COMP_LEGEND       = 56
FS_COMP_LEGEND_TITLE = 62

# ── line / marker sizes ───────────────────────────────────────────────
MS_IND       = 11.6
LW_IND_DATA  = 3.2
LW_IND_TREND = 3.6
LW_IND_GRID  = 1.0
LW_IND_SPINE = 6.0

MS_COMP       = 7.0
LW_COMP_DATA  = 2.0
LW_COMP_TREND = 3.6
LW_COMP_GRID  = 1.0
LW_COMP_SPINE = 9.0

# ── biome colours and labels ──────────────────────────────────────────
BIOME_COLORS = {
     1: "#007f00",  2: "#7fc241",  3: "#004f00",  4: "#86aa7a",
     5: "#005b6f",  6: "#77a5b8",  7: "#f4d942",  8: "#d5d064",
     9: "#9fd0db", 10: "#9cbf6b", 12: "#1b8e55",
    13: "#e74a2d", 14: "#00936f",
}
BIOME_NAMES = {
     1: "Tropical Moist Broadleaf Forests",
     2: "Tropical Dry Broadleaf Forests",
     3: "Tropical Coniferous Forests",
     4: "Temperate Broadleaf & Mixed Forests",
     5: "Temperate Conifer Forests",
     6: "Boreal Forests/Taiga",
     7: "Tropical Grasslands & Savannas",
     8: "Temperate Grasslands & Savannas",
     9: "Flooded Grasslands & Savannas",
    10: "Montane Grasslands & Shrublands",
    12: "Mediterranean Forests, Woodlands & Scrub",
    13: "Deserts & Xeric Shrublands",
    14: "Mangroves",
}

VALID_BIOMES    = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14]
EXCLUDED_BIOMES = [11]

TOP_PANELS    = [1, 2, 3]
LEFT_PANELS   = [4, 5, 6, 7]
RIGHT_PANELS  = [8, 9, 10]
BOTTOM_PANELS = [12, 13, 14]

# ── helpers ───────────────────────────────────────────────────────────
def _save_multi(fig, stem: Path, dpi: int | None = None):
    stem = Path(stem)
    fig.savefig(stem.with_suffix(".png"), dpi=dpi or DPI,
                bbox_inches="tight", facecolor="white")
    if SAVE_PDF:
        fig.savefig(stem.with_suffix(".pdf"),
                    bbox_inches="tight", facecolor="white")
    if SAVE_SVG:
        fig.savefig(stem.with_suffix(".svg"),
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)

def _should_clean(biome_id: int) -> bool:
    if CLEAN_MODE == "all":  return True
    if CLEAN_MODE == "none": return False
    return biome_id in CLEAN_SUBSET

def mad_outlier_mask(y: np.ndarray, k: float = MAD_K) -> np.ndarray:
    y = np.asarray(y, dtype="float64")
    if y.size < 3:
        return np.zeros_like(y, dtype=bool)
    dy  = np.diff(y)
    med = np.nanmedian(dy)
    mad = np.nanmedian(np.abs(dy - med))
    if not np.isfinite(mad) or mad == 0:
        return np.zeros_like(y, dtype=bool)
    z    = np.abs(dy - med) / (1.4826 * mad)
    mask = np.zeros_like(y, dtype=bool)
    mask[1:] = z > k
    return mask

def clean_series(y: np.ndarray, k: float = MAD_K) -> np.ndarray:
    s = pd.Series(np.asarray(y, dtype="float64"))
    m = mad_outlier_mask(s.values, k=k)
    if m.any():
        s.loc[m] = np.nan
        s = s.interpolate(limit_direction="both")
    s = s.rolling(window=3, center=True, min_periods=1).median()
    return s.values

def _apply_sci_y(ax):
    if PLOT_AS_MKM2:
        fmt = ScalarFormatter(useMathText=False)
        fmt.set_scientific(False)
        ax.yaxis.set_major_formatter(fmt)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.yaxis.offsetText.set_visible(False)
    else:
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.offsetText.set_visible(True)

def _remove_antarctica(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf[gdf.geometry.centroid.y > -60].copy()
    return gdf

def _y_for_plot(y_km2: np.ndarray) -> tuple[np.ndarray, str]:
    if PLOT_AS_MKM2:
        return y_km2 / 1e6, "Area, million km²"
    return y_km2, "Area, km²"

def _one_annotation_text(slopes: pd.DataFrame, bid: int) -> str:
    s_km2yr = float(slopes.at[bid, "slope_km2_per_year"])
    y0      = float(slopes.at[bid, "y0_km2"])
    annpct  = slopes.at[bid, "annual_pct_rate"]
    perM    = slopes.at[bid, "slope_per_Mkm2"]
    fullpct = float(slopes.at[bid, "pct_change_full_period"])

    if ONLY_TEXT == "annual_pct_rate":
        val = float(annpct) if np.isfinite(annpct) else (
            100.0 * s_km2yr / y0 if np.isfinite(y0) and y0 != 0 else np.nan)
        return f"{val:+.3f} %/yr"
    if ONLY_TEXT == "slope_km2yr":
        return f"{s_km2yr:+,.0f} km²/yr"
    if ONLY_TEXT == "slope_per_Mkm2":
        val = float(perM) if np.isfinite(perM) else (
            s_km2yr / (y0 / 1e6) if np.isfinite(y0) and y0 != 0 else np.nan)
        return f"{val:+.1f} km²/yr/Mkm²"
    return f"{fullpct:+.1f} %"

def _pick_corner(text_corner: str, bid: int) -> str:
    if text_corner != "random":
        return text_corner
    rng = np.random.default_rng(bid) if RANDOM_CORNER_DETERMINISTIC else np.random.default_rng()
    return rng.choice(["tl", "tr", "bl", "br"])

def _percent_limits(series: pd.Series):
    s = series.dropna().astype(float)
    if PCT_RANGE_MODE == "fixed" or s.empty:
        return PCT_VMIN, PCT_VMAX
    lo, hi = np.percentile(s, [5, 95])
    m = max(abs(lo), abs(hi))
    m = float(np.ceil(m / 5.0) * 5.0)
    return -m, m

# ── ENSO monthly loader (shared) ──────────────────────────────────────
def _load_enso_monthly() -> pd.Series:
    ds    = xr.open_dataset(ENSO_FILE, decode_times=False)
    dates = pd.date_range(start=ENSO_START, periods=len(ds["time"]), freq="MS")
    return pd.Series(ds[ENSO_VAR].values.ravel(), index=dates, name="oni")

# ── ENSO cross-correlation computation ───────────────────────────────
def _corr_at_lag(enso_s: pd.Series, area_s: pd.Series, lag: int) -> float:
    """Pearson r between enso_s and area_s shifted by -lag (ENSO leads area)."""
    y    = area_s.shift(-lag)
    mask = enso_s.notna() & y.notna()
    if mask.sum() < MIN_N:
        return np.nan
    return float(np.corrcoef(enso_s[mask].values, y[mask].values)[0, 1])

def compute_enso_correlations() -> dict:
    """
    Returns dict with keys 'ElNino' and 'LaNina', each a GeoDataFrame
    with columns 'region', 'corr', 'lag', 'geometry' ready for plotting.
    """
    print("Loading realm area data for ENSO analysis...")
    df_r = pd.read_parquet(AREA_REALM_FILE)
    df_r["time"]     = pd.to_datetime(df_r["time"], format="%b-%Y")
    df_r["biome_id"] = df_r["BIOME_ID_REALM"].str.extract(r"^(\d+)").astype(int)
    df_r = df_r[df_r["biome_id"] != 11]
    df_r = df_r[~df_r["BIOME_ID_REALM"].str.endswith("_AN")]
    area = df_r.pivot(index="time", columns="BIOME_ID_REALM",
                      values="area_km2").sort_index()

    monthly = _load_enso_monthly()
    phase   = pd.Series(
        np.select([monthly > 0.5, monthly < -0.5], ["ElNino", "LaNina"], "Neutral"),
        index=monthly.index,
    )

    # load and dissolve shapefile by biome+realm region
    print("Loading biome-realm shapefile...")
    gdf_shp = gpd.read_file(BIOME_SHP)[["BIOME", "REALM", "geometry"]]
    gdf_shp = gdf_shp[gdf_shp["BIOME"].between(1, 14) & (gdf_shp["BIOME"] != 11)]
    gdf_shp = gdf_shp[~gdf_shp["REALM"].eq("AN")]
    gdf_shp = gdf_shp.to_crs("EPSG:4326")
    gdf_shp["region"] = (gdf_shp["BIOME"].astype(int).astype(str)
                         + "_" + gdf_shp["REALM"])
    gdf_shp = gdf_shp.dissolve(by="region", as_index=False)
    gdf_shp["geometry"] = gdf_shp["geometry"].buffer(0)
    if REMOVE_ANTARCTICA:
        gdf_shp = gdf_shp[
            gdf_shp.geometry.centroid.y > -60
        ].copy()

    results = {}
    for ph in ["ElNino", "LaNina"]:
        idx    = phase[phase == ph].index
        en_ph  = monthly.reindex(idx)
        records = []
        for region in area.columns:
            ar = area[region].reindex(idx)
            rs = []
            for lag in range(0, MAX_LAG + 1):
                r = _corr_at_lag(en_ph, ar, lag)
                if np.isfinite(r):
                    rs.append((lag, r))
            if not rs:
                continue
            lag_star, r_star = max(rs, key=lambda t: abs(t[1]))
            if abs(r_star) >= CORR_MIN:
                records.append({"region": region,
                                 "corr":   r_star,
                                 "lag":    lag_star})

        df_ph = pd.DataFrame(records)
        if df_ph.empty:
            results[ph] = gdf_shp.copy()
            results[ph]["corr"] = np.nan
            results[ph]["lag"]  = np.nan
        else:
            results[ph] = gdf_shp.merge(df_ph, on="region", how="left")

        print(f"  {ph}: {len(df_ph)} regions with |r| >= {CORR_MIN}")

    return results

# ── ENSO spatial map helper ───────────────────────────────────────────
def _iter_geoms(geom):
    if geom is None or geom.is_empty:
        return
    if isinstance(geom, __import__("shapely").geometry.MultiPolygon):
        for g in geom.geoms:
            if not g.is_empty:
                yield g
    elif isinstance(geom, __import__("shapely").geometry.Polygon):
        yield geom
    else:
        g2 = geom.buffer(0)
        if not g2.is_empty:
            yield from _iter_geoms(g2)

def _robinson_enso_map(gdf_plot, value_col, cmap_name,
                        vmin, vmax, title, stem,
                        integer_ticks=False, cb_label=""):
    """
    Single Robinson projection map in the same style as the pct-change map.
    Saves PNG, PDF, SVG at DPI resolution.
    """
    fig, ax = plt.subplots(
        figsize=(31, 15.6),
        subplot_kw={"projection": ccrs.Robinson()},
        dpi=DPI,
        facecolor="white",
    )
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
    ax.coastlines(resolution="110m", lw=0.6, color="0.45", zorder=1)
    ax.axis("off")

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)

    for _, row in gdf_plot.iterrows():
        val = row[value_col]
        if not np.isfinite(val):
            continue
        for poly in _iter_geoms(row.geometry):
            ax.add_geometries(
                [poly],
                crs=ccrs.PlateCarree(),
                facecolor=cmap(norm(val)),
                edgecolor="none",
                linewidth=0,
                zorder=2,
            )

    ax.set_extent([-180, 180, -58, 90], crs=ccrs.PlateCarree())
    ax.set_title(title, fontsize=FS_ENSO_MAP_TITLE,
                 fontweight="bold", pad=16)

    # colorbar — same placement style as pct map
    cax = inset_axes(
        ax,
        width="1.8%", height="50%",
        loc="lower left",
        bbox_to_anchor=(-0.08, 0.25, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, cax=cax, orientation="vertical")
    cb.outline.set_visible(False)
    if integer_ticks:
        cb.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    else:
        cb.set_ticks(np.linspace(vmin, vmax, 5))
    cb.set_label(cb_label, labelpad=6, fontsize=FS_ENSO_MAP_CB)
    cb.ax.tick_params(labelsize=FS_ENSO_MAP_CB)

    _save_multi(fig, Path(OUTPUT_FOLDER) / stem)
    print(f"  Saved {stem}")

def plot_enso_spatial_maps():
    print("Computing ENSO biome-realm correlations...")
    results = compute_enso_correlations()

    # correlation maps — RdBu_r, -1 to 1
    for ph, label in [("ElNino", "El Niño"), ("LaNina", "La Niña")]:
        _robinson_enso_map(
            gdf_plot     = results[ph],
            value_col    = "corr",
            cmap_name    = "RdBu_r",
            vmin=-1, vmax=1,
            title        = f"Max ENSO–wetGDE correlation — {label} (1979–2019)",
            stem         = f"enso_corr_{ph}",
            integer_ticks= False,
            cb_label     = "Pearson r",
        )

    # lag maps — Oranges, 0 to MAX_LAG
    for ph, label in [("ElNino", "El Niño"), ("LaNina", "La Niña")]:
        _robinson_enso_map(
            gdf_plot     = results[ph],
            value_col    = "lag",
            cmap_name    = "Oranges",
            vmin=0, vmax=MAX_LAG,
            title        = f"Lag of max ENSO–wetGDE correlation — {label} (months)",
            stem         = f"enso_lag_{ph}",
            integer_ticks= True,
            cb_label     = "Lag (months)",
        )

# ── data I/O ──────────────────────────────────────────────────────────
def load_monthly_area() -> pd.DataFrame:
    df = pd.read_parquet(AREA_FILE)
    df["time"] = pd.to_datetime(df["time"], format="%b-%Y")
    df = df[df["biome_id"].isin(VALID_BIOMES)].copy()
    print(f"Loaded monthly data: {df.shape}")
    return df

def annual_means(df: pd.DataFrame) -> pd.DataFrame:
    ann = (
        df.set_index("time")
          .groupby("biome_id")["area_km2"]
          .resample("YE")
          .apply(lambda s: s.mean() if s.notna().sum() >= 6 else np.nan)
          .reset_index()
    )
    print(f"Computed annual means: {ann.shape}")
    return ann

def load_biome_map() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(BIOME_SHP)[["BIOME", "geometry"]]
    gdf = gdf.rename(columns={"BIOME": "biome_id"}).astype({"biome_id": "int"})
    gdf = gdf[gdf["biome_id"].isin(VALID_BIOMES)].copy()
    if REMOVE_ANTARCTICA:
        gdf = _remove_antarctica(gdf)
    return gdf

# ── metrics ───────────────────────────────────────────────────────────
def compute_biome_trends(df_ann: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bid, grp in df_ann.groupby("biome_id"):
        if bid not in VALID_BIOMES:
            continue
        print(f"Computing trend for biome {bid}...")
        yrs   = grp["time"].dt.year.to_numpy(dtype=float)
        y     = grp["area_km2"].to_numpy(dtype=float)
        y_use = clean_series(y) if _should_clean(bid) else y

        m = np.isfinite(yrs) & np.isfinite(y_use)
        if m.sum() > 1:
            s, b, _, _ = theilslopes(y_use[m], yrs[m])
            y0             = b + s * yrs[m].min()
            yN             = b + s * yrs[m].max()
            dA             = yN - y0
            pct            = (dA / y0 * 100.0) if np.isfinite(y0) and y0 != 0 else np.nan
            annual_pct     = (100.0 * s / y0)  if np.isfinite(y0) and y0 != 0 else np.nan
            slope_per_Mkm2 = (s / (y0 / 1e6))  if np.isfinite(y0) and y0 != 0 else np.nan
        else:
            s = y0 = yN = dA = pct = annual_pct = slope_per_Mkm2 = np.nan

        rows.append({
            "biome_id":               bid,
            "label":                  BIOME_NAMES.get(bid, f"Biome {bid}"),
            "slope_km2_per_year":     s,
            "pct_change_full_period": pct,
            "y0_km2":                 y0,
            "yN_km2":                 yN,
            "delta_km2":              dA,
            "annual_pct_rate":        annual_pct,
            "slope_per_Mkm2":         slope_per_Mkm2,
        })

    out = pd.DataFrame(rows).set_index("biome_id")
    total_base        = out["y0_km2"].sum(skipna=True)
    out["pp_contrib"] = 100.0 * out["delta_km2"] / total_base if total_base != 0 else np.nan
    return out

def global_annual_series(df_mon: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        df_mon.groupby("time", as_index=False)["area_km2"]
              .sum(min_count=1)
              .sort_values("time")
    )
    ann = (
        monthly.set_index("time")["area_km2"]
               .resample("YE")
               .apply(lambda s: s.mean() if s.notna().sum() >= 6 else np.nan)
               .reset_index()
    )
    return ann

# ── plotting primitive ────────────────────────────────────────────────
def draw_trend_panel(
    ax, grp, slopes, bid,
    title_fs, tick_fs, ann_fs, ylabel_fs,
    show_ylabel=True, xtick_step=5,
    spine_lw=LW_IND_SPINE, ann_color=None,
    ms=MS_IND, lw_data=LW_IND_DATA,
    lw_trend=LW_IND_TREND, lw_grid=LW_IND_GRID,
):
    yrs     = grp["time"].dt.year.to_numpy()
    y       = grp["area_km2"].to_numpy(dtype=float)
    y_clean = clean_series(y) if _should_clean(bid) else y
    y_plot, y_label = _y_for_plot(y_clean)

    finite = np.isfinite(y_plot)
    if finite.sum() > 0:
        first_idx = np.where(finite)[0][0]
        rel       = y_plot - y_plot[first_idx]
        rel[~finite] = np.nan
        drop_yr = yrs[np.nanargmin(rel)]
        rise_yr = yrs[np.nanargmax(rel)]
    else:
        drop_yr = rise_yr = yrs[0]

    ax.axvspan(drop_yr - 0.5, drop_yr + 0.5, facecolor="lightgrey", alpha=0.30, zorder=0)
    ax.axvspan(rise_yr - 0.5, rise_yr + 0.5, facecolor="lightgrey", alpha=0.30, zorder=0)
    ax.plot(yrs, y_plot, "--o", ms=ms, lw=lw_data, color="dodgerblue")

    m = np.isfinite(yrs) & np.isfinite(y_plot)
    if m.sum() > 1:
        s_fit, b_fit, _, _ = theilslopes(y_plot[m], yrs[m])
        ax.plot(yrs[m], b_fit + s_fit * yrs[m], "--", lw=lw_trend, color="tomato")

    ax.set_title(f"Biome {bid}", fontsize=title_fs, fontweight="bold", pad=6)
    if show_ylabel:
        ax.set_ylabel(y_label, fontsize=ylabel_fs)
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelleft=False)

    ax.grid(True, ls=":", lw=lw_grid, color="0.72")
    ax.tick_params(axis="x", rotation=45, labelsize=tick_fs, length=0)
    ax.tick_params(axis="y", labelsize=tick_fs, length=0)
    ax.set_xticks(np.arange(yrs.min(), yrs.max() + 1, xtick_step))
    _apply_sci_y(ax)

    txt    = _one_annotation_text(slopes, bid)
    corner = _pick_corner(TEXT_CORNER, bid)
    x      = 0.02 if corner.endswith("l") else 0.98
    y_pos  = 0.95 if corner.startswith("t") else 0.05
    ha     = "left"  if corner.endswith("l") else "right"
    va     = "top"   if corner.startswith("t") else "bottom"
    ax.text(x, y_pos, txt, transform=ax.transAxes, ha=ha, va=va,
            fontsize=ann_fs, fontweight="bold",
            color=ann_color if ann_color is not None else "black")

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(spine_lw)
        spine.set_edgecolor(BIOME_COLORS.get(bid, "black"))

# ── Global time-series panel (below pct map) ──────────────────────────
def _draw_global_ts_panel(ax, df_mon: pd.DataFrame):
    """
    Draws global annual mean GDE area with Theil-Sen linear trend and
    3-year centred rolling mean, matching the reference figure style.
    """
    g      = global_annual_series(df_mon)
    years  = g["time"].dt.year.to_numpy()
    y      = g["area_km2"].to_numpy(dtype=float)

    # 3-year centred rolling mean
    roll3 = (
        pd.Series(y, index=years)
          .rolling(window=3, center=True, min_periods=1)
          .mean()
          .to_numpy()
    )

    # Theil-Sen linear trend (consistent with biome panels)
    m = np.isfinite(years.astype(float)) & np.isfinite(y)
    if m.sum() > 1:
        s_fit, b_fit, _, _ = theilslopes(y[m], years[m].astype(float))
        trend_y = b_fit + s_fit * years.astype(float)
    else:
        trend_y = np.full_like(y, np.nan)

    # ── plot ──
    ax.plot(years, y,       "-o",  color="#4c9be8", lw=1.8, ms=5,
            label="Annual global mean", zorder=3)
    ax.plot(years, trend_y, "--",  color="black",   lw=2.0,
            label="Linear trend",       zorder=4)
    ax.plot(years, roll3,   "-",   color="#e8a020", lw=2.2,
            label="3 yr rolling mean",  zorder=5)

    ax.set_ylabel("Area, km²", fontsize=FS_PCT_MAP_TS_AX)
    ax.set_xlabel("",           fontsize=FS_PCT_MAP_TS_AX)
    ax.tick_params(labelsize=FS_PCT_MAP_TS_AX, length=0, axis="both")

    # x-ticks every 5 years
    xticks = np.arange((years.min() // 5) * 5, years.max() + 1, 5)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(yr) for yr in xticks], rotation=45, ha="right")
    ax.set_xlim(years.min() - 0.5, years.max() + 0.5)

    ax.grid(True, ls=":", lw=0.6, color="0.75", zorder=0)

    # scientific notation on y-axis (×10⁶ style)
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((6, 6))
    ax.yaxis.set_major_formatter(fmt)

    # legend floats above the axes, centred, no frame
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=3,
        frameon=False,
        fontsize=FS_PCT_MAP_TS_LEG,
        handlelength=2.0,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

# ── individual trend plots ────────────────────────────────────────────
def plot_trend_panels(df_ann: pd.DataFrame, slopes: pd.DataFrame):
    for bid in VALID_BIOMES:
        print(f"Saving individual trend plot for biome {bid}...")
        grp = df_ann[df_ann["biome_id"] == bid].sort_values("time")
        fig, ax = plt.subplots(figsize=(21, 15), dpi=DPI)
        draw_trend_panel(ax, grp, slopes, bid,
                         FS_TREND_TITLE, FS_TREND_TICK, FS_TREND_ANN, FS_TREND_YLABEL,
                         show_ylabel=True, xtick_step=5,
                         spine_lw=LW_IND_SPINE, ann_color=None,
                         ms=MS_IND, lw_data=LW_IND_DATA,
                         lw_trend=LW_IND_TREND, lw_grid=LW_IND_GRID)
        fig.tight_layout()
        _save_multi(fig, Path(OUTPUT_FOLDER) / f"trend_biome_{bid:02d}")

# ── pct-change map ────────────────────────────────────────────────────
def plot_pct_map(slopes: pd.DataFrame, df_mon: pd.DataFrame):
    """
    Full-period % change choropleth with a global time-series panel below.
    """
    print("Saving full-period pct-change map...")
    gdf = load_biome_map()
    df  = gdf.set_index("biome_id").join(slopes[["pct_change_full_period"]])
    vmin, vmax = _percent_limits(df["pct_change_full_period"])

    fig = plt.figure(figsize=(31, 22), dpi=DPI, facecolor="white")
    gs  = mgridspec.GridSpec(
        2, 1,
        height_ratios=[3.2, 1.0],
        hspace=0.28,
    )
    ax    = fig.add_subplot(gs[0], projection=ccrs.Robinson())
    ax_ts = fig.add_subplot(gs[1])

    # ── choropleth map ──
    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
    ax.coastlines(resolution="110m", lw=0.6, color="0.45", zorder=1)
    df.plot(
        column="pct_change_full_period", cmap="RdBu",
        vmin=vmin, vmax=vmax,
        linewidth=0, edgecolor="none",
        alpha=0.95, transform=ccrs.PlateCarree(),
        ax=ax, zorder=2,
    )
    ax.set_extent([-180, 180, -58, 90], crs=ccrs.PlateCarree())
    ax.axis("off")

    cax = inset_axes(
        ax,
        width=PCT_CB_WIDTH, height=PCT_CB_HEIGHT,
        loc="lower left",
        bbox_to_anchor=PCT_CB_BBOX,
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    sm = plt.cm.ScalarMappable(
        cmap="RdBu",
        norm=plt.Normalize(vmin=vmin, vmax=vmax),
    )
    cb = fig.colorbar(sm, cax=cax, orientation="vertical")
    cb.outline.set_visible(False)
    cb.set_ticks(np.linspace(vmin, vmax, 5))
    cb.set_label("% change", labelpad=6, fontsize=FS_PCT_MAP_CB)
    cb.ax.tick_params(labelsize=FS_PCT_MAP_CB)

    # ── global time-series panel ──
    _draw_global_ts_panel(ax_ts, df_mon)

    _save_multi(fig, Path(OUTPUT_FOLDER) / "pct_change_full_period_map")

# ── legend handles ────────────────────────────────────────────────────
def make_legend_handles():
    handles = [
        Patch(facecolor=BIOME_COLORS[b], edgecolor="none",
              label=f"{b} – {BIOME_NAMES[b]}")
        for b in VALID_BIOMES
    ]
    handles.append(Patch(facecolor="#888888", edgecolor="none", label="Other / Masked"))
    return handles

# ── composite ─────────────────────────────────────────────────────────
def make_composite(df_ann: pd.DataFrame, slopes: pd.DataFrame):
    print("Saving composite figure...")
    gdf = load_biome_map()
    gdf["color"] = gdf["biome_id"].map(BIOME_COLORS)

    fig = plt.figure(figsize=(64, 52), dpi=COMPOSITE_DPI, facecolor="white")
    gs  = fig.add_gridspec(
        nrows=7, ncols=7,
        height_ratios=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.5],
        width_ratios=[1.0, 0.08, 1.0, 1.0, 1.0, 0.08, 1.0],
        hspace=0.60, wspace=0.32,
    )

    for bid, col in zip(TOP_PANELS, [2, 3, 4]):
        ax  = fig.add_subplot(gs[0, col])
        grp = df_ann[df_ann["biome_id"] == bid].sort_values("time")
        draw_trend_panel(ax, grp, slopes, bid,
                         FS_COMP_PANEL_TITLE, FS_COMP_PANEL_TICK,
                         FS_COMP_PANEL_ANN,   FS_COMP_PANEL_YLABEL,
                         show_ylabel=False, xtick_step=10,
                         spine_lw=LW_COMP_SPINE, ann_color=BIOME_COLORS.get(bid),
                         ms=MS_COMP, lw_data=LW_COMP_DATA,
                         lw_trend=LW_COMP_TREND, lw_grid=LW_COMP_GRID)

    for bid, row in zip(LEFT_PANELS, [1, 2, 3, 4]):
        ax  = fig.add_subplot(gs[row, 0])
        grp = df_ann[df_ann["biome_id"] == bid].sort_values("time")
        draw_trend_panel(ax, grp, slopes, bid,
                         FS_COMP_PANEL_TITLE, FS_COMP_PANEL_TICK,
                         FS_COMP_PANEL_ANN,   FS_COMP_PANEL_YLABEL,
                         show_ylabel=False, xtick_step=10,
                         spine_lw=LW_COMP_SPINE, ann_color=BIOME_COLORS.get(bid),
                         ms=MS_COMP, lw_data=LW_COMP_DATA,
                         lw_trend=LW_COMP_TREND, lw_grid=LW_COMP_GRID)

    for bid, row in zip(RIGHT_PANELS, [1, 2, 3]):
        ax  = fig.add_subplot(gs[row, 6])
        grp = df_ann[df_ann["biome_id"] == bid].sort_values("time")
        draw_trend_panel(ax, grp, slopes, bid,
                         FS_COMP_PANEL_TITLE, FS_COMP_PANEL_TICK,
                         FS_COMP_PANEL_ANN,   FS_COMP_PANEL_YLABEL,
                         show_ylabel=False, xtick_step=10,
                         spine_lw=LW_COMP_SPINE, ann_color=BIOME_COLORS.get(bid),
                         ms=MS_COMP, lw_data=LW_COMP_DATA,
                         lw_trend=LW_COMP_TREND, lw_grid=LW_COMP_GRID)

    for bid, col in zip(BOTTOM_PANELS, [2, 3, 4]):
        ax  = fig.add_subplot(gs[5, col])
        grp = df_ann[df_ann["biome_id"] == bid].sort_values("time")
        draw_trend_panel(ax, grp, slopes, bid,
                         FS_COMP_PANEL_TITLE, FS_COMP_PANEL_TICK,
                         FS_COMP_PANEL_ANN,   FS_COMP_PANEL_YLABEL,
                         show_ylabel=False, xtick_step=10,
                         spine_lw=LW_COMP_SPINE, ann_color=BIOME_COLORS.get(bid),
                         ms=MS_COMP, lw_data=LW_COMP_DATA,
                         lw_trend=LW_COMP_TREND, lw_grid=LW_COMP_GRID)

    ax_map = fig.add_subplot(gs[1:5, 1:6], projection=ccrs.Robinson())
    ax_map.add_feature(cfeature.OCEAN, facecolor="#d6eaf8", zorder=0)
    ax_map.add_feature(cfeature.LAND,  facecolor="white",   zorder=1)
    ax_map.coastlines(resolution="110m", lw=1.0, color="0.55", zorder=2)
    gdf.plot(ax=ax_map, color=gdf["color"], linewidth=0, edgecolor="none",
             transform=ccrs.PlateCarree(), zorder=3)
    ax_map.set_extent([-180, 180, -58, 90], crs=ccrs.PlateCarree())
    ax_map.axis("off")

    ax_leg = fig.add_subplot(gs[6, :])
    ax_leg.axis("off")
    ax_leg.legend(handles=make_legend_handles(), loc="center", ncol=3,
                  frameon=False, title="Biome ID – Name",
                  fontsize=FS_COMP_LEGEND, title_fontsize=FS_COMP_LEGEND_TITLE,
                  handlelength=1.8, handleheight=1.4,
                  borderpad=0.8, labelspacing=1.0, columnspacing=3.0)

    _save_multi(fig, Path(OUTPUT_FOLDER) / "composite_biome_map_with_trends",
                dpi=COMPOSITE_DPI)

# ── summaries ─────────────────────────────────────────────────────────
def save_summary_csv(slopes: pd.DataFrame):
    out = Path(OUTPUT_FOLDER) / "biome_trend_summary.csv"
    slopes.to_csv(out, float_format="%.6g")
    print(f"Saved biome summary CSV: {out}")

def save_global_summary_csv(df_mon: pd.DataFrame):
    g     = global_annual_series(df_mon)
    years = g["time"].dt.year.to_numpy()
    y     = g["area_km2"].to_numpy()
    m     = np.isfinite(years) & np.isfinite(y)
    if m.sum() > 1:
        s, b, _, _ = theilslopes(y[m], years[m])
        y0         = b + s * years[m].min()
        yN         = b + s * years[m].max()
        pct_full   = (yN - y0) / y0 * 100 if np.isfinite(y0) and y0 != 0 else np.nan
        annual_pct = 100.0 * s / y0        if np.isfinite(y0) and y0 != 0 else np.nan
    else:
        s = y0 = yN = pct_full = annual_pct = np.nan

    out = Path(OUTPUT_FOLDER) / "global_trend_summary.csv"
    pd.DataFrame([{
        "slope_km2_per_year":       s,
        "steady_loss_km2_per_year": -min(s, 0.0) if np.isfinite(s) else np.nan,
        "y0_km2": y0, "yN_km2": yN,
        "pct_change_full_period": pct_full,
        "annual_pct_rate": annual_pct,
    }]).to_csv(out, index=False, float_format="%.6g")
    print(f"Saved global summary CSV: {out}")

# ── main ──────────────────────────────────────────────────────────────
def main():
    t0 = time.time()

    print("Loading monthly data...")
    df_mon = load_monthly_area()

    print("Aggregating to annual means...")
    df_ann = annual_means(df_mon)

    print("Computing biome trends...")
    slopes = compute_biome_trends(df_ann)

    if MAKE_TRENDS:
        plot_trend_panels(df_ann, slopes)

    if MAKE_PCT_MAP:
        plot_pct_map(slopes, df_mon)

    if MAKE_COMPOSITE:
        make_composite(df_ann, slopes)

    if MAKE_SUMMARIES:
        save_summary_csv(slopes)
        save_global_summary_csv(df_mon)

    if MAKE_ENSO_SPATIAL:
        print("Saving ENSO spatial maps...")
        plot_enso_spatial_maps()

    print(f"Saved outputs to {OUTPUT_FOLDER}")
    print(f"Done in {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()