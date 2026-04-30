"""
prepare_pixel_overlays.py
Reads the wetGDE climatology (30-arcsec) and persistence mask (5-arcmin),
reprojects persistence mask to the wetGDE grid, computes pixel-level metrics,
and writes GeoTIFFs + coloured PNGs for the dashboard.

All outputs are at 30-arcsec resolution matching the wetGDE extent.

Run on Snellius:
    python prepare_pixel_overlays.py

Outputs written to ./overlays/:
    persist_class.tif / .png
    wet_season_length.tif / .png
    dry_stress.tif / .png
    risk_pixel.tif / .png
    monthly_MM.tif / .png  (01-12)
    meta_overlays.json
"""

import json
import warnings
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from affine import Affine
import xarray as xr
from PIL import Image

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────
PERSIST_TIF = Path(
    "/gpfs/scratch1/shared/otoo0001/paper_2/output_gde/"
    "persistence_mask/wetgde_persistence_mask_historical.tif"
)
WET_NC = Path(
    "/gpfs/scratch1/shared/otoo0001/past_gde/wetGDE_clim_2015_2019.nc"
)
OUT_DIR = Path("overlays")
OUT_DIR.mkdir(exist_ok=True)

MONTH_ABBR = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]

# ── Colours ───────────────────────────────────────────────────
COL_EPISODIC  = (254, 196,  79, 210)
COL_SEASONAL  = ( 65, 171,  93, 220)
COL_PERENNIAL = ( 37,  52, 148, 240)

WET_LEN_CMAP = [
    (255,255,178),(254,217,142),(254,178, 76),(253,141, 60),(240, 59, 32),
    (189,  0, 38),(161,218,180),( 65,182,196),( 44,127,184),( 37, 52,148),
    (  8, 29, 88),(  0,  0, 50),
]
STRESS_CMAP = [
    (255,255,255),(254,224,210),(252,187,161),(252,146,114),(251,106, 74),
    (239, 59, 44),(203, 24, 29),(165, 15, 21),(103,  0, 13),
]
RISK_CMAP = [
    ( 26,152, 80),(102,189, 99),(166,217,106),(217,239,139),(255,255,191),
    (254,224,139),(253,174, 97),(244,109, 67),(215, 48, 39),(165,  0, 38),
]

# ── Helpers ───────────────────────────────────────────────────
def apply_cmap(data, cmap, vmin, vmax, nodata_mask):
    n    = len(cmap)
    norm = np.clip((data - vmin) / max(vmax - vmin, 1e-9), 0, 1)
    idx  = np.minimum((norm * (n-1)).astype(int), n-1)
    rgba = np.zeros((*data.shape, 4), dtype=np.uint8)
    for i, col in enumerate(cmap):
        mask = (idx == i) & ~nodata_mask
        rgba[mask] = (*col, 210)
    rgba[nodata_mask] = (0, 0, 0, 0)
    return rgba

def save_png(rgba, path):
    Image.fromarray(rgba, mode="RGBA").save(path, "PNG", optimize=True)
    print(f"  {path.name}  {rgba.shape[1]}x{rgba.shape[0]}  "
          f"{path.stat().st_size//1024} KB")

def save_tif(data, path, transform, nodata=-1):
    with rasterio.open(
        path, "w", driver="GTiff",
        height=data.shape[0], width=data.shape[1],
        count=1, dtype=data.dtype,
        crs="EPSG:4326", transform=transform,
        compress="lzw", nodata=nodata,
    ) as dst:
        dst.write(data, 1)
    print(f"  {path.name}  {data.shape[1]}x{data.shape[0]}  "
          f"{path.stat().st_size//1024} KB")

# ── Load wetGDE grid ──────────────────────────────────────────
print("Loading wetGDE climatology...")
ds = xr.open_dataset(WET_NC, decode_times=False)
if "latitude" in ds.coords:
    ds = ds.rename({"latitude": "lat", "longitude": "lon"})

lat = ds["lat"].values
lon = ds["lon"].values
height, width = len(lat), len(lon)
lat_res = abs(float(lat[1] - lat[0]))
lon_res = abs(float(lon[1] - lon[0]))

# rasterio transform for the wetGDE grid (north-first)
dst_transform = Affine(
     lon_res, 0, float(lon.min()) - lon_res / 2,
     0, -lat_res, float(lat.max()) + lat_res / 2,
)

# Leaflet bounds [[south,west],[north,east]]
south = float(lat.min()) - lat_res / 2
north = float(lat.max()) + lat_res / 2
west  = float(lon.min()) - lon_res / 2
east  = float(lon.max()) + lon_res / 2
LEAFLET_BOUNDS = [[south, west], [north, east]]

print(f"  wetGDE shape: {height}x{width}")
print(f"  Resolution:   {lat_res:.6f}°")
print(f"  Bounds:       S={south:.4f} W={west:.4f} N={north:.4f} E={east:.4f}")

wet_arr = np.nan_to_num(ds["wetGDE"].values, nan=0.0).astype(np.float32)
ds.close()

# if lat is descending, flip so row 0 = southernmost (for numpy indexing)
# but keep north-first for GeoTIFF writing
lat_desc = lat[0] > lat[-1]
if not lat_desc:
    wet_arr = wet_arr[:, ::-1, :]   # flip lat axis
    lat     = lat[::-1]
# now lat[0] = max (north), consistent with dst_transform

# ── Reproject persistence mask to wetGDE grid ─────────────────
print("\nReprojecting persistence mask to wetGDE grid...")
persist = np.full((height, width), -1, dtype=np.int8)

with rasterio.open(PERSIST_TIF) as src:
    reproject(
        source=rasterio.band(src, 1),
        destination=persist,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=dst_transform,
        dst_crs="EPSG:4326",
        dst_nodata=-1,
        resampling=Resampling.nearest,
    )

print(f"  persist shape: {persist.shape}")
print(f"  unique values: {np.unique(persist).tolist()}")
print(f"  class counts:  { {int(v): int((persist==v).sum()) for v in np.unique(persist)} }")

nodata_mask    = persist == -1
gde_mask       = persist > 0
episodic_mask  = persist == 1
seasonal_mask  = persist == 2
perennial_mask = persist == 3

# ── Number of wet months per pixel ───────────────────────────
n_wet = wet_arr.sum(axis=0)  # (H, W)

# ── Cell area (varies by latitude) ───────────────────────────
lat_rad      = np.deg2rad(lat)
cell_area_1d = (
    np.abs(np.sin(lat_rad + np.deg2rad(lat_res/2)) -
           np.sin(lat_rad - np.deg2rad(lat_res/2)))
    * np.deg2rad(lon_res) * 6371.0**2
)
cell_area_2d = np.repeat(cell_area_1d[:, None], width, axis=1)

# ── 1. Persistence class overlay ─────────────────────────────
print("\n[1/5] Persistence class...")
rgba = np.zeros((height, width, 4), dtype=np.uint8)
rgba[episodic_mask]  = COL_EPISODIC
rgba[seasonal_mask]  = COL_SEASONAL
rgba[perennial_mask] = COL_PERENNIAL
save_png(rgba, OUT_DIR / "persist_class.png")
save_tif(persist.copy(), OUT_DIR / "persist_class.tif", dst_transform)

# ── 2. Wet season length ──────────────────────────────────────
print("\n[2/5] Wet season length...")
wet_nodata = (n_wet == 0) | nodata_mask
rgba = apply_cmap(n_wet, WET_LEN_CMAP, 1, 12, wet_nodata)
save_png(rgba, OUT_DIR / "wet_season_length.png")
n_wet_int = n_wet.astype(np.int8)
n_wet_int[nodata_mask] = -1
save_tif(n_wet_int, OUT_DIR / "wet_season_length.tif", dst_transform)

# ── 3. Dry stress ─────────────────────────────────────────────
print("\n[3/5] Dry season stress...")
with np.errstate(divide="ignore", invalid="ignore"):
    mean_wet   = n_wet / 12.0
    min_wet    = wet_arr.min(axis=0)
    dry_stress = np.where(mean_wet > 0, 1.0 - (min_wet / mean_wet), 0.0)

stress_nodata = nodata_mask | ~gde_mask
rgba = apply_cmap(dry_stress, STRESS_CMAP, 0, 1, stress_nodata)
save_png(rgba, OUT_DIR / "dry_stress.png")
dry_int = np.where(stress_nodata, np.int8(-1),
                   (dry_stress * 100).clip(-127, 127).astype(np.int8))
save_tif(dry_int, OUT_DIR / "dry_stress.tif", dst_transform)

# ── 4. Pixel risk ─────────────────────────────────────────────
print("\n[4/5] Pixel risk...")
persist_norm = np.clip(persist.astype(float) / 3.0, 0, 1)
instability  = 1.0 - np.clip(n_wet / 12.0, 0, 1)
risk_pixel   = 0.4 * instability + 0.4 * dry_stress + 0.2 * persist_norm
risk_nodata  = nodata_mask | ~gde_mask
rgba = apply_cmap(risk_pixel, RISK_CMAP, 0, 1, risk_nodata)
save_png(rgba, OUT_DIR / "risk_pixel.png")
risk_int = np.where(risk_nodata, np.int8(-1),
                    (risk_pixel * 100).clip(-127, 127).astype(np.int8))
save_tif(risk_int, OUT_DIR / "risk_pixel.tif", dst_transform)

# ── 5. Monthly overlays ───────────────────────────────────────
print("\n[5/5] Monthly presence (persistence-coloured)...")
monthly_stats = []

for m in range(12):
    month_wet = wet_arr[m] > 0
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[month_wet & perennial_mask] = COL_PERENNIAL
    rgba[month_wet & seasonal_mask]  = COL_SEASONAL
    rgba[month_wet & episodic_mask]  = COL_EPISODIC
    save_png(rgba, OUT_DIR / f"monthly_{m+1:02d}.png")

    # GeoTIFF: persistence class where wet, 0 where dry, -1 nodata
    month_tif = np.where(nodata_mask,           np.int8(-1),
                np.where(month_wet & perennial_mask, np.int8(3),
                np.where(month_wet & seasonal_mask,  np.int8(2),
                np.where(month_wet & episodic_mask,  np.int8(1),
                                                     np.int8(0)))))
    save_tif(month_tif, OUT_DIR / f"monthly_{m+1:02d}.tif", dst_transform)

    # stats
    wet_this = month_wet & ~nodata_mask
    area_t = float(cell_area_2d[wet_this].sum())
    area_p = float(cell_area_2d[wet_this & perennial_mask].sum())
    area_s = float(cell_area_2d[wet_this & seasonal_mask].sum())
    area_e = float(cell_area_2d[wet_this & episodic_mask].sum())
    monthly_stats.append({
        "month": m+1, "abbr": MONTH_ABBR[m],
        "area_km2":           round(area_t, 1),
        "area_perennial_km2": round(area_p, 1),
        "area_seasonal_km2":  round(area_s, 1),
        "area_episodic_km2":  round(area_e, 1),
        "n_pixels":           int(wet_this.sum()),
    })
    print(f"  {MONTH_ABBR[m]}: {area_t:,.0f} km²  "
          f"(peren={area_p:,.0f}  seas={area_s:,.0f}  epis={area_e:,.0f})")

# ── Write meta ────────────────────────────────────────────────
print("\nWriting meta_overlays.json...")
meta = {
    "bounds":         {"west":west,"south":south,"east":east,"north":north},
    "leaflet_bounds": LEAFLET_BOUNDS,
    "layers": [
        {
            "id":"persist_class","label":"Persistence Class",
            "file":"overlays/persist_class.tif",
            "legend":[
                {"label":"Episodic",  "color":"#fec44f"},
                {"label":"Seasonal",  "color":"#41ab5d"},
                {"label":"Perennial", "color":"#253494"},
            ],
        },
        {
            "id":"wet_season_length","label":"Wet Season Length (months)",
            "file":"overlays/wet_season_length.tif",
            "legend":[
                {"label":"1 month",   "color":"#ffffb2"},
                {"label":"6 months",  "color":"#fd8d3c"},
                {"label":"12 months", "color":"#000032"},
            ],
        },
        {
            "id":"dry_stress","label":"Dry Season Stress",
            "file":"overlays/dry_stress.tif",
            "legend":[
                {"label":"Low",  "color":"#ffffff"},
                {"label":"High", "color":"#67000d"},
            ],
        },
        {
            "id":"risk_pixel","label":"Pixel Risk Index",
            "file":"overlays/risk_pixel.tif",
            "legend":[
                {"label":"Low",      "color":"#1a9850"},
                {"label":"Moderate", "color":"#ffffbf"},
                {"label":"High",     "color":"#a50026"},
            ],
        },
        {
            "id":"monthly","label":"Monthly GDW Presence","animated":True,
            "files":[f"overlays/monthly_{m+1:02d}.tif" for m in range(12)],
            "months":MONTH_ABBR,
            "monthly_stats":monthly_stats,
            "legend":[
                {"label":"Episodic",  "color":"#fec44f"},
                {"label":"Seasonal",  "color":"#41ab5d"},
                {"label":"Perennial", "color":"#253494"},
            ],
        },
    ],
    "pixel_stats":{
        "n_episodic":  int(episodic_mask.sum()),
        "n_seasonal":  int(seasonal_mask.sum()),
        "n_perennial": int(perennial_mask.sum()),
        "n_total_gde": int(gde_mask.sum()),
    },
}

with open(OUT_DIR / "meta_overlays.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nDone. All outputs in ./overlays/")
print(f"  Bounds: S={south:.4f} W={west:.4f} N={north:.4f} E={east:.4f}")
print(f"  Episodic:  {int(episodic_mask.sum()):,} pixels")
print(f"  Seasonal:  {int(seasonal_mask.sum()):,} pixels")
print(f"  Perennial: {int(perennial_mask.sum()):,} pixels")