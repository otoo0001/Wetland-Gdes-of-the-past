#!/usr/bin/env python3
"""
plot_wetgde_persistence_mask_robinson.py
----------------------------------------
Loads an already saved wetGDE persistence mask GeoTIFF and creates a
Robinson-projection PNG with a legend. Open water (GLWD classes 1-7)
is masked out before plotting.

Classes in the input GeoTIFF:
    -1 - NoData / excluded
     0 - Non-GDE
     1 - Episodic GDE
     2 - Seasonal GDE
     3 - Perennial GDE
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import cartopy.crs as ccrs

# ── Configuration ─────────────────────────────────────────────────────────────

IN_TIF   = "/gpfs/scratch1/shared/otoo0001/paper_2/revisions/shapefiles_from/wetgde_persistence_mask_historical_2015_2019.tif"
GLWD_TIF = "/gpfs/scratch1/shared/otoo0001/from_projects/validation/input_files/glwd/GLWD_v2_delta_combined_classes/GLWD_v2_delta_main_class.tif"
OUT_PNG  = "/gpfs/scratch1/shared/otoo0001/paper_2/revisions/figures/persistence/wetgde_persistence_mask_historical_robinson.png"

TITLE          = ""
OPEN_WATER_CLS = set(range(1, 8))  # GLWD classes 1-7

print("Configuration loaded.")
print(f"  Input persistence mask : {IN_TIF}")
print(f"  Input GLWD TIF         : {GLWD_TIF}")
print(f"  Output PNG             : {OUT_PNG}")

# ── Load persistence mask ─────────────────────────────────────────────────────

print("\nLoading persistence mask...")
with rasterio.open(IN_TIF) as src:
    mask         = src.read(1).astype(float)
    bounds       = src.bounds
    height       = src.height
    width        = src.width
    nodata_value = src.nodata if src.nodata is not None else -1
    print(f"  Shape      : {height} x {width}")
    print(f"  Bounds     : {bounds}")
    print(f"  NoData val : {nodata_value}")
    print(f"  Unique values (raw): {np.unique(mask[~np.isnan(mask)]).astype(int)}")

mask[mask == nodata_value] = np.nan
print(f"  NoData pixels set to NaN. Valid pixels: {np.sum(~np.isnan(mask)):,}")

# ── Load and resample GLWD, mask open water ───────────────────────────────────

print("\nLoading and resampling GLWD open-water mask...")
with rasterio.open(GLWD_TIF) as gsrc:
    glwd = gsrc.read(
        1,
        out_shape=(height, width),
        resampling=Resampling.nearest
    ).astype(float)
    glwd_nodata = gsrc.nodata
    if glwd_nodata is not None:
        glwd[glwd == glwd_nodata] = np.nan
    print(f"  Resampled to {height} x {width}")
    print(f"  GLWD NoData val: {glwd_nodata}")
    print(f"  GLWD unique classes (excl. NaN): {np.unique(glwd[~np.isnan(glwd)]).astype(int)}")

is_open_water = np.isin(glwd, list(OPEN_WATER_CLS))
n_masked = int(np.sum(is_open_water))
print(f"  Open-water pixels to mask (GLWD classes 1-7): {n_masked:,}")
mask[is_open_water] = np.nan
print(f"  Valid pixels after open-water masking: {np.sum(~np.isnan(mask)):,}")

# class counts
for cls, label in [(1, "Episodic"), (2, "Seasonal"), (3, "Perennial")]:
    count = int(np.sum(mask == cls))
    print(f"  Class {cls} ({label}): {count:,} pixels")

# ── Pixel center coordinates ──────────────────────────────────────────────────

print("\nComputing pixel centre coordinates...")
lon = np.linspace(
    bounds.left  + (bounds.right - bounds.left) / (2 * width),
    bounds.right - (bounds.right - bounds.left) / (2 * width),
    width,
)
lat = np.linspace(
    bounds.top    - (bounds.top - bounds.bottom) / (2 * height),
    bounds.bottom + (bounds.top - bounds.bottom) / (2 * height),
    height,
)

lon2d, lat2d = np.meshgrid(lon, lat)
print(f"  Lon range: {lon.min():.3f} to {lon.max():.3f}")
print(f"  Lat range: {lat.min():.3f} to {lat.max():.3f}")

# ── Styling ───────────────────────────────────────────────────────────────────

print("\nSetting up colormap and legend...")
cmap = ListedColormap([
    "#ffffff",  # 0 Non-GDE
    "#F0E442",  # 1 Episodic
    "#0072B2",  # 2 Seasonal
    "#D55E00",  # 3 Perennial
])
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

legend_handles = [
    Patch(facecolor="#F0E442", edgecolor="none", label="Episodic (1-3 months/year)"),
    Patch(facecolor="#0072B2", edgecolor="none", label="Seasonal (3-6 months/year)"),
    Patch(facecolor="#D55E00", edgecolor="none", label="Perennial (>6 months/year)"),
]
print("  Colormap and legend handles ready.")

# ── Plot ──────────────────────────────────────────────────────────────────────

print("\nCreating figure (Robinson projection)...")
fig = plt.figure(figsize=(16, 8), dpi=300)
ax  = plt.axes(projection=ccrs.Robinson())
ax.set_global()
print("  Figure and axes initialised.")

print("  Rendering pcolormesh (this may take a moment at full resolution)...")
ax.pcolormesh(
    lon2d, lat2d, mask,
    cmap=cmap,
    norm=norm,
    transform=ccrs.PlateCarree(),
    shading="auto",
    rasterized=True,
)
print("  pcolormesh rendered.")

print("  Adding coastlines...")
ax.coastlines(linewidth=0.3, color="black")

print("  Adding legend...")
ax.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.05),
    ncol=3,
    frameon=False,
    fontsize=12,
)

ax.set_axis_off()

print("\nSaving figure...")
Path(OUT_PNG).parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()

out_png = Path(OUT_PNG)
out_pdf = out_png.with_suffix(".pdf")

plt.savefig(out_png, bbox_inches="tight", facecolor="white", dpi=300)
print(f"  PNG saved : {out_png}")

plt.savefig(out_pdf, bbox_inches="tight", facecolor="white")
print(f"  PDF saved : {out_pdf}")

plt.close(fig)
print("Done.")