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

IN_TIF   = "/gpfs/scratch1/shared/otoo0001/paper_2/revisions/shapefiles_from/wetgde_max_presence_mask_2015_2019.tif"
GLWD_TIF = "/gpfs/scratch1/shared/otoo0001/from_projects/validation/input_files/glwd/GLWD_v2_delta_combined_classes/GLWD_v2_delta_main_class.tif"
OUT_PNG  = "/gpfs/scratch1/shared/otoo0001/paper_2/revisions/figures/persistence/wetgde_persistence_mask_historical_robinson.png"

TITLE          = ""
OPEN_WATER_CLS = set(range(1, 8))  # GLWD classes 1-7

# ── Load persistence mask ─────────────────────────────────────────────────────

with rasterio.open(IN_TIF) as src:
    mask         = src.read(1).astype(float)
    bounds       = src.bounds
    height       = src.height
    width        = src.width
    nodata_value = src.nodata if src.nodata is not None else -1

mask[mask == nodata_value] = np.nan

# ── Load and resample GLWD, mask open water ───────────────────────────────────

with rasterio.open(GLWD_TIF) as gsrc:
    glwd = gsrc.read(
        1,
        out_shape=(height, width),
        resampling=Resampling.nearest
    ).astype(float)
    glwd_nodata = gsrc.nodata
    if glwd_nodata is not None:
        glwd[glwd == glwd_nodata] = np.nan

is_open_water = np.isin(glwd, list(OPEN_WATER_CLS))
mask[is_open_water] = np.nan

# ── Pixel center coordinates ──────────────────────────────────────────────────

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

# ── Styling ───────────────────────────────────────────────────────────────────

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

# ── Plot ──────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 8), dpi=300)
ax  = plt.axes(projection=ccrs.Robinson())
ax.set_global()

ax.pcolormesh(
    lon2d, lat2d, mask,
    cmap=cmap,
    norm=norm,
    transform=ccrs.PlateCarree(),
    shading="auto",
    rasterized=True,
)
ax.coastlines(linewidth=0.3, color="black")

ax.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.05),
    ncol=3,
    frameon=False,
    fontsize=12,
)

ax.set_axis_off()

Path(OUT_PNG).parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.show()
plt.savefig(OUT_PNG, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"Saved: {OUT_PNG}")