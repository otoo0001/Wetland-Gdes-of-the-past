#!/usr/bin/env python3

from pathlib import Path
from collections import defaultdict
import csv

import numpy as np
import rasterio
import rasterio.features
from rasterio.warp import reproject
from rasterio.enums import Resampling
from rasterio.transform import xy
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from scipy.ndimage import distance_transform_edt


# =============================================================================
# INPUTS
# =============================================================================
INTERSECTION_TIF = Path(
    "/scratch-shared/otoo0001/from_projects/validation/input_files/rhodes_2024_gde/Rhodes/"
    "glwd_rhodes_intersection_for_validation/"
    "glwd_rhodes_intersection_binary_no_open.tif"
)

PERSISTENCE_TIF = Path(
    "/scratch-shared/otoo0001/paper_2/revisions/shapefiles_from/"
    "wetgde_max_presence_mask_2015_2019.tif"
)

DRYLAND_MASK_TIF = Path(
    "/scratch-shared/otoo0001/from_projects/validation/input_files/koppen_geiger/1991_2020/"
    "koppen_geiger_0p5.tif"
)

GLWD_CLASS_TIF = Path(
    "/scratch-shared/otoo0001/from_projects/validation/input_files/glwd/"
    "GLWD_v2_delta_combined_classes/GLWD_v2_delta_main_class.tif"
)

OUTDIR = Path(
    "/scratch-shared/otoo0001/paper_2/revisions/figures/"
    "evaluation_glwd_rohdes_10KM"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

PERSISTENCE_NODATA = -1
GLWD_NODATA = 0

INTERSECTION_POSITIVE = 1
OPEN_WATER_VALUE = 2
COMBINED_CLASSES = [1, 2, 3]
PERSISTENCE_CLASS_LABELS = {
    1: "Episodic",
    2: "Seasonal",
    3: "Perennial",
}
DRYLAND_CLASSES = {4, 5, 6, 7, 8, 9, 10}

PAD = 200
TOLERANCE_KM = 5.0
MAX_PLOT_ROWS = 1800
MAX_PLOT_COLS = 3600
TILE_SIZE = 1200

GLWD_CLASS_NAMES = {
    1: "Class 1",
    2: "Class 2",
    3: "Class 3",
    4: "Class 4",
    5: "Class 5",
    6: "Class 6",
    7: "Class 7",
    8: "Class 8",
    9: "Class 9",
    10: "Class 10",
    11: "Class 11",
    12: "Class 12",
}


# =============================================================================
# HELPERS
# =============================================================================
def load_raster(path):
    with rasterio.open(path) as src:
        arr = src.read(1)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
    return arr, profile, transform, crs, nodata


def align_to_grid(
    src_arr,
    src_transform,
    src_crs,
    src_nodata,
    dst_shape,
    dst_transform,
    dst_crs,
    dst_nodata,
    resampling=Resampling.nearest,
):
    out = np.full(dst_shape, dst_nodata, dtype=np.float32)

    reproject(
        source=src_arr,
        destination=out,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=src_nodata,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=dst_nodata,
        dst_width=dst_shape[1],
        dst_height=dst_shape[0],
        resampling=resampling,
    )
    return out


def write_dict_txt(path, d):
    with open(path, "w") as f:
        for k, v in d.items():
            f.write(f"{k}: {v}\n")


def write_metrics_txt(path, metrics):
    with open(path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")


def write_missed_glwd_summary(path, summary, code_to_continent, glwd_class_names):
    with open(path, "w") as f:
        grand_total = 0

        for cont_code in sorted(summary):
            cont_name = code_to_continent.get(cont_code, f"Continent_{cont_code}")
            if cont_name == "Antarctica":
                continue

            cont_total = sum(summary[cont_code].values())
            grand_total += cont_total

            f.write(f"{cont_name}\n")
            f.write(f"total_missed_reference_pixels: {cont_total}\n")

            for glwd_code in sorted(summary[cont_code]):
                count = summary[cont_code][glwd_code]
                frac = count / cont_total if cont_total > 0 else np.nan
                glwd_name = glwd_class_names.get(glwd_code, f"Class {glwd_code}")
                f.write(
                    f"  glwd_class_{glwd_code} ({glwd_name}): "
                    f"{count} ({frac:.4f})\n"
                )

            f.write("\n")

        f.write(f"grand_total_missed_reference_pixels: {grand_total}\n")


def downsample_for_plot(arr, max_rows=MAX_PLOT_ROWS, max_cols=MAX_PLOT_COLS):
    nrows, ncols = arr.shape
    row_step = max(1, int(np.ceil(nrows / max_rows)))
    col_step = max(1, int(np.ceil(ncols / max_cols)))
    return arr[::row_step, ::col_step], row_step, col_step


def get_bbox_from_mask(mask, pad=0):
    rows, cols = np.where(mask)
    if rows.size == 0 or cols.size == 0:
        raise RuntimeError("No positive pixels found for requested mask.")

    r0 = max(0, rows.min() - pad)
    r1 = min(mask.shape[0], rows.max() + pad + 1)
    c0 = max(0, cols.min() - pad)
    c1 = min(mask.shape[1], cols.max() + pad + 1)
    return r0, r1, c0, c1


def crop_extent(transform, r0, r1, c0, c1):
    west, north = xy(transform, r0, c0, offset="ul")
    east, south = xy(transform, r1 - 1, c1 - 1, offset="lr")
    return [west, east, south, north]


def rgba_from_class_array(arr, class_to_rgba, nodata_value=255):
    out = np.zeros(arr.shape + (4,), dtype=np.uint8)
    for cls, rgba in class_to_rgba.items():
        out[arr == cls] = rgba
    out[arr == nodata_value] = (255, 255, 255, 0)
    return out


def add_map_base(ax):
    ax.set_global()
    ax.add_feature(
        cfeature.COASTLINE.with_scale("110m"),
        linewidth=0.4,
        edgecolor="0.3",
    )
    ax.set_axis_off()
    ax.set_frame_on(False)


def get_pixel_sizes_km(transform, extent):
    lon_res_deg = abs(transform.a)
    lat_res_deg = abs(transform.e)
    center_lat = 0.5 * (extent[2] + extent[3])

    row_km = lat_res_deg * 111.32
    col_km = lon_res_deg * 111.32 * max(np.cos(np.deg2rad(center_lat)), 1e-6)

    return row_km, col_km, center_lat


def get_tolerance_pad_pixels(transform, extent, tolerance_km):
    row_km, col_km, _ = get_pixel_sizes_km(transform, extent)
    pad_rows = int(np.ceil(tolerance_km / row_km)) + 2
    pad_cols = int(np.ceil(tolerance_km / col_km)) + 2
    return pad_rows, pad_cols, row_km, col_km


def build_continent_raster(dst_shape, dst_transform):
    continent_names = [
        "Africa",
        "Asia",
        "Europe",
        "North America",
        "South America",
        "Oceania",
        "Antarctica",
    ]
    continent_to_code = {name: i + 1 for i, name in enumerate(continent_names)}
    code_to_continent = {v: k for k, v in continent_to_code.items()}

    shp = shpreader.natural_earth(
        resolution="110m",
        category="cultural",
        name="admin_0_countries",
    )

    shapes = []
    for rec in shpreader.Reader(shp).records():
        cont = rec.attributes["CONTINENT"]
        if cont in continent_to_code:
            shapes.append((rec.geometry, continent_to_code[cont]))

    continent_raster = rasterio.features.rasterize(
        shapes,
        out_shape=dst_shape,
        transform=dst_transform,
        fill=0,
        dtype="uint8",
    )

    return continent_raster, code_to_continent


def summarize_continent_glwd(mask, glwd_arr, continent_arr, glwd_nodata=0):
    summary = defaultdict(lambda: defaultdict(int))

    valid = (
        mask
        & np.isfinite(glwd_arr)
        & np.isfinite(continent_arr)
        & (glwd_arr != glwd_nodata)
        & (continent_arr != 0)
    )

    if valid.sum() == 0:
        return summary

    cont_vals = continent_arr[valid].astype(np.int16)
    glwd_vals = glwd_arr[valid].astype(np.int16)

    for cont, glwd in zip(cont_vals, glwd_vals):
        summary[int(cont)][int(glwd)] += 1

    return summary


def subtract_nested_counts(total_summary, subtract_summary):
    out = defaultdict(lambda: defaultdict(int))

    continent_keys = set(total_summary.keys()) | set(subtract_summary.keys())
    for cont in continent_keys:
        class_keys = set(total_summary.get(cont, {}).keys()) | set(
            subtract_summary.get(cont, {}).keys()
        )
        for cls in class_keys:
            total_val = total_summary.get(cont, {}).get(cls, 0)
            sub_val = subtract_summary.get(cont, {}).get(cls, 0)
            diff = total_val - sub_val
            if diff < 0:
                diff = 0
            if diff > 0:
                out[cont][cls] = diff

    return out


def build_downsampled_plot_array(
    ref_full,
    pred_full,
    tp_pred_full,
    fp_full,
    hit_ref_full,
    max_rows=MAX_PLOT_ROWS,
    max_cols=MAX_PLOT_COLS,
):
    arr_ref, row_step, col_step = downsample_for_plot(
        ref_full, max_rows=max_rows, max_cols=max_cols,
    )

    tp_pred_plot = tp_pred_full[::row_step, ::col_step]
    fp_plot = fp_full[::row_step, ::col_step]
    hit_ref_plot = hit_ref_full[::row_step, ::col_step]

    plot = np.full(arr_ref.shape, 255, dtype=np.uint8)
    fn_plot = arr_ref & (~hit_ref_plot)

    plot[fp_plot] = 1
    plot[fn_plot] = 2
    plot[tp_pred_plot] = 3

    return plot, row_step, col_step


def plot_tp_fp_downsampled(paths, arr_plot, extent, row_step, col_step):
    """Spatial TP/FP/FN map, standalone."""
    class_to_rgba = {
        1: (217, 95, 2, 255),
        2: (160, 160, 160, 255),
        3: (27, 158, 119, 255),
    }
    rgba = rgba_from_class_array(arr_plot, class_to_rgba, nodata_value=255)
    rgba[(arr_plot != 1) & (arr_plot != 2) & (arr_plot != 3)] = (255, 255, 255, 0)

    fig = plt.figure(figsize=(33, 20))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())

    ax.imshow(
        rgba,
        origin="upper",
        extent=extent,
        transform=ccrs.PlateCarree(),
        interpolation="nearest",
        regrid_shape=3600,
    )
    add_map_base(ax)

    handles = [
        Patch(facecolor=np.array(class_to_rgba[3]) / 255.0, edgecolor="none", label="True Positive (TP)"),
        Patch(facecolor=np.array(class_to_rgba[1]) / 255.0, edgecolor="none", label="False Positive (FP)"),
        Patch(facecolor=np.array(class_to_rgba[2]) / 255.0, edgecolor="none", label="False Negative (FN)"),
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=3,
        frameon=False,
        fontsize=40,
    )

    plt.tight_layout()
    for path in paths:
        if str(path).lower().endswith(".pdf"):
            plt.savefig(path, bbox_inches="tight", pad_inches=0.15)
        else:
            plt.savefig(path, dpi=1200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def _draw_barplot(ax, records_plot, legend_anchor_y=-0.32):
    """Shared bar chart drawing logic for standalone and combined figures."""
    continents = [r["continent"] for r in records_plot]
    glwd_vals = np.array([r["glwd_area_km2"] for r in records_plot], dtype=float)
    rhodes_vals = np.array([r["rhodes_glwd_intersection_area_km2"] for r in records_plot], dtype=float)
    sim_vals = np.array([r["simulated_classes123_area_km2"] for r in records_plot], dtype=float)

    scale, xlabel = get_area_scale_and_label(
        max(np.nanmax(glwd_vals), np.nanmax(rhodes_vals), np.nanmax(sim_vals))
    )

    color_glwd = "#cfd4dc"
    color_rhodes = "#1f4e79"
    color_sim_edge = "#d95f02"
    color_sim_fill = "#fdd0a2"

    y = np.arange(len(continents))
    bar_h = 0.62
    inner_h = 0.36

    ax.barh(y, glwd_vals / scale, height=bar_h, color=color_glwd, edgecolor="none", zorder=1)
    ax.barh(y, rhodes_vals / scale, height=inner_h, color=color_rhodes, edgecolor="none", zorder=3)
    ax.barh(
        y,
        sim_vals / scale,
        height=inner_h * 0.82,
        color=color_sim_fill,
        edgecolor=color_sim_edge,
        linewidth=2.0,
        zorder=4,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(continents, fontsize=40)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=40)
    ax.grid(axis="x", color="0.88", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.ticklabel_format(style="plain", axis="x", useOffset=False)
    ax.tick_params(axis="x", labelsize=40)

    handles = [
        Patch(facecolor=color_glwd, edgecolor="none",
              label="Total wetlands (GLWDv2) in drylands"),
        Patch(facecolor=color_rhodes, edgecolor="none",
              label="Groundwater dependent wetlands in drylands \u2229 GLWDv2  (Rohde et al.)"),
        Patch(facecolor=color_sim_fill, edgecolor=color_sim_edge, linewidth=2.0,
              label="Simulated groundwater dependent wetlands in drylands (this study)"),
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5,-0.6),
        ncol=1,
        fontsize=40,
    )


def plot_area_barplot(paths, records):
    records_plot = [r for r in records if r["continent"] != "Antarctica"]
    records_plot = records_plot + [_build_global_record(records_plot)]

    fig, ax = plt.subplots(figsize=(22, 16))
    _draw_barplot(ax, records_plot)

    plt.tight_layout(rect=[0, 0.10, 1, 1])
    for path in paths:
        if str(path).lower().endswith(".pdf"):
            plt.savefig(path, bbox_inches="tight")
        else:
            plt.savefig(path, dpi=1200, bbox_inches="tight")
    plt.close(fig)


def _build_global_record(records_plot):
    """Aggregate continent records into a single Global record."""
    global_rec = {
        "continent": "Global",
        "glwd_area_km2": float(sum(r["glwd_area_km2"] for r in records_plot)),
        "rhodes_glwd_intersection_area_km2": float(
            sum(r["rhodes_glwd_intersection_area_km2"] for r in records_plot)
        ),
        "simulated_classes123_area_km2": float(
            sum(r["simulated_classes123_area_km2"] for r in records_plot)
        ),
    }
    global_rec["intersection_over_glwd"] = (
        global_rec["rhodes_glwd_intersection_area_km2"] / global_rec["glwd_area_km2"]
        if global_rec["glwd_area_km2"] > 0 else np.nan
    )
    global_rec["simulated_over_glwd"] = (
        global_rec["simulated_classes123_area_km2"] / global_rec["glwd_area_km2"]
        if global_rec["glwd_area_km2"] > 0 else np.nan
    )
    global_rec["simulated_over_intersection"] = (
        global_rec["simulated_classes123_area_km2"]
        / global_rec["rhodes_glwd_intersection_area_km2"]
        if global_rec["rhodes_glwd_intersection_area_km2"] > 0 else np.nan
    )
    return global_rec


def plot_combined_figure(paths, arr_plot, extent, records):
    """
    Two-panel publication figure:
      top    -- Robinson projection TP/FP/FN map  (panel a)
      bottom -- continent-wise area bar chart      (panel b)
    """
    records_plot = [r for r in records if r["continent"] != "Antarctica"]
    records_plot = records_plot + [_build_global_record(records_plot)]

    class_to_rgba = {
        1: (217, 95, 2, 255),
        2: (160, 160, 160, 255),
        3: (27, 158, 119, 255),
    }
    rgba = rgba_from_class_array(arr_plot, class_to_rgba, nodata_value=255)
    rgba[(arr_plot != 1) & (arr_plot != 2) & (arr_plot != 3)] = (255, 255, 255, 0)

    fig = plt.figure(figsize=(30, 40))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1], hspace=-0.30)

    ax_map = fig.add_subplot(gs[0], projection=ccrs.Robinson())
    ax_map.imshow(
        rgba,
        origin="upper",
        extent=extent,
        transform=ccrs.PlateCarree(),
        interpolation="nearest",
        regrid_shape=3600,
    )
    add_map_base(ax_map)

    map_handles = [
        Patch(facecolor=np.array(class_to_rgba[3]) / 255.0, edgecolor="none", label="True Positive (TP)"),
        Patch(facecolor=np.array(class_to_rgba[1]) / 255.0, edgecolor="none", label="False Positive (FP)"),
        Patch(facecolor=np.array(class_to_rgba[2]) / 255.0, edgecolor="none", label="False Negative (FN)"),
    ]
    ax_map.legend(
        handles=map_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=False,
        fontsize=40,
    )
    ax_map.text(
        -0.02, 1.02, "(a)",
        transform=ax_map.transAxes,
        fontsize=40, fontweight="bold", va="top", ha="left",
    )

    ax_bar = fig.add_subplot(gs[1])
    _draw_barplot(ax_bar, records_plot, legend_anchor_y=-0.52)
    ax_bar.text(
        -0.08, 1.08, "(b)",
        transform=ax_bar.transAxes,
        fontsize=40, fontweight="bold", va="top", ha="left",
    )

    for path in paths:
        if str(path).lower().endswith(".pdf"):
            plt.savefig(path, bbox_inches="tight", pad_inches=0.2)
        else:
            plt.savefig(path, dpi=1200, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def iter_tiles(nrows, ncols, tile_size):
    for rr0 in range(0, nrows, tile_size):
        rr1 = min(nrows, rr0 + tile_size)
        for cc0 in range(0, ncols, tile_size):
            cc1 = min(ncols, cc0 + tile_size)
            yield rr0, rr1, cc0, cc1


def evaluate_combined_tiled(
    pred,
    ref,
    glwd_crop,
    continent_crop,
    inter_transform,
    extent_crop,
    crop_row_offset,
    crop_col_offset,
    tolerance_km,
    reference_positive_count,
    reference_summary,
):
    predicted_positive_count = int(pred.sum())

    crop_row_km, crop_col_km, crop_center_lat = get_pixel_sizes_km(
        inter_transform, extent_crop,
    )
    pad_rows_tol, pad_cols_tol, _, _ = get_tolerance_pad_pixels(
        inter_transform, extent_crop, tolerance_km,
    )

    if predicted_positive_count == 0:
        metrics = {
            "predicted_label": "classes_1_to_3_combined",
            "allowable_distance_km": float(tolerance_km),
            "cropped_row_start": int(crop_row_offset),
            "cropped_row_end": int(crop_row_offset + ref.shape[0]),
            "cropped_col_start": int(crop_col_offset),
            "cropped_col_end": int(crop_col_offset + ref.shape[1]),
            "cropped_height": int(ref.shape[0]),
            "cropped_width": int(ref.shape[1]),
            "dryland_classes": sorted(DRYLAND_CLASSES),
            "open_water_value_excluded": int(OPEN_WATER_VALUE),
            "combined_classes": COMBINED_CLASSES,
            "valid_pixels": int(ref.size),
            "reference_positive_pixels": int(reference_positive_count),
            "predicted_positive_pixels": 0,
            "tolerant_true_positive_pred_pixels": 0,
            "tolerant_false_alarm_pixels": 0,
            "tolerant_reference_hits": 0,
            "tolerant_reference_misses": int(reference_positive_count),
            "precision": float(np.nan),
            "recall": float(0.0 if reference_positive_count > 0 else np.nan),
            "f1": float(np.nan),
            "approx_row_pixel_size_km": float(crop_row_km),
            "approx_col_pixel_size_km": float(crop_col_km),
            "crop_center_latitude": float(crop_center_lat),
            "tile_size": int(TILE_SIZE),
            "pad_rows_tolerance": int(pad_rows_tol),
            "pad_cols_tolerance": int(pad_cols_tol),
        }

        missed_summary = reference_summary
        tp_pred_mask = np.zeros(ref.shape, dtype=bool)
        fp_mask = np.zeros(ref.shape, dtype=bool)
        hit_ref_mask = np.zeros(ref.shape, dtype=bool)

        return metrics, missed_summary, tp_pred_mask, fp_mask, hit_ref_mask

    dist_to_ref = distance_transform_edt(
        ~ref, sampling=(crop_row_km, crop_col_km),
    )

    tp_pred_mask = pred & (dist_to_ref <= tolerance_km)
    fp_mask = pred & (dist_to_ref > tolerance_km)

    tp_pred_count = int(tp_pred_mask.sum())
    fp_count = int(fp_mask.sum())

    hit_ref_mask = np.zeros(ref.shape, dtype=bool)
    nrows, ncols = pred.shape

    for tr0, tr1, tc0, tc1 in iter_tiles(nrows, ncols, TILE_SIZE):
        hr0 = max(0, tr0 - pad_rows_tol)
        hr1 = min(nrows, tr1 + pad_rows_tol)
        hc0 = max(0, tc0 - pad_cols_tol)
        hc1 = min(ncols, tc1 + pad_cols_tol)

        pred_halo = pred[hr0:hr1, hc0:hc1]
        if not pred_halo.any():
            continue

        ref_halo = ref[hr0:hr1, hc0:hc1]

        halo_extent = crop_extent(
            inter_transform,
            crop_row_offset + hr0,
            crop_row_offset + hr1,
            crop_col_offset + hc0,
            crop_col_offset + hc1,
        )
        row_km_h, col_km_h, _ = get_pixel_sizes_km(inter_transform, halo_extent)

        dist_to_pred_halo = distance_transform_edt(
            ~pred_halo, sampling=(row_km_h, col_km_h),
        )

        core_r0 = tr0 - hr0
        core_r1 = core_r0 + (tr1 - tr0)
        core_c0 = tc0 - hc0
        core_c1 = core_c0 + (tc1 - tc0)

        hit_core = (
            ref_halo[core_r0:core_r1, core_c0:core_c1]
            & (dist_to_pred_halo[core_r0:core_r1, core_c0:core_c1] <= tolerance_km)
        )

        hit_ref_mask[tr0:tr1, tc0:tc1] |= hit_core

        del pred_halo, ref_halo, dist_to_pred_halo, hit_core

    hit_ref_count = int(hit_ref_mask.sum())
    fn_count = int(reference_positive_count - hit_ref_count)

    precision = (
        tp_pred_count / predicted_positive_count
        if predicted_positive_count > 0 else np.nan
    )
    recall = (
        hit_ref_count / reference_positive_count
        if reference_positive_count > 0 else np.nan
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
        else np.nan
    )

    metrics = {
        "predicted_label": "classes_1_to_3_combined",
        "allowable_distance_km": float(tolerance_km),
        "cropped_row_start": int(crop_row_offset),
        "cropped_row_end": int(crop_row_offset + ref.shape[0]),
        "cropped_col_start": int(crop_col_offset),
        "cropped_col_end": int(crop_col_offset + ref.shape[1]),
        "cropped_height": int(ref.shape[0]),
        "cropped_width": int(ref.shape[1]),
        "dryland_classes": sorted(DRYLAND_CLASSES),
        "open_water_value_excluded": int(OPEN_WATER_VALUE),
        "combined_classes": COMBINED_CLASSES,
        "valid_pixels": int(ref.size),
        "reference_positive_pixels": int(reference_positive_count),
        "predicted_positive_pixels": int(predicted_positive_count),
        "tolerant_true_positive_pred_pixels": int(tp_pred_count),
        "tolerant_false_alarm_pixels": int(fp_count),
        "tolerant_reference_hits": int(hit_ref_count),
        "tolerant_reference_misses": int(fn_count),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "approx_row_pixel_size_km": float(crop_row_km),
        "approx_col_pixel_size_km": float(crop_col_km),
        "crop_center_latitude": float(crop_center_lat),
        "tile_size": int(TILE_SIZE),
        "pad_rows_tolerance": int(pad_rows_tol),
        "pad_cols_tolerance": int(pad_cols_tol),
    }

    hit_summary = summarize_continent_glwd(
        hit_ref_mask, glwd_crop, continent_crop, glwd_nodata=GLWD_NODATA,
    )
    missed_summary = subtract_nested_counts(reference_summary, hit_summary)

    del dist_to_ref, hit_summary

    return metrics, missed_summary, tp_pred_mask, fp_mask, hit_ref_mask


def get_row_area_km2(transform, nrows):
    lat_res_deg = abs(transform.e)
    lon_res_deg = abs(transform.a)

    row_areas = np.zeros(nrows, dtype=np.float64)

    for r in range(nrows):
        _, lat_center = xy(transform, r, 0, offset="center")
        pixel_height_km = lat_res_deg * 111.32
        pixel_width_km = lon_res_deg * 111.32 * max(
            np.cos(np.deg2rad(lat_center)), 1e-12,
        )
        row_areas[r] = pixel_height_km * pixel_width_km

    return row_areas


def area_by_continent_km2(mask, continent_arr, row_areas_km2, code_to_continent):
    out = {}
    valid_conts = sorted(np.unique(continent_arr[continent_arr > 0]).astype(int))

    for cont_code in valid_conts:
        cont_name = code_to_continent.get(cont_code, f"Continent_{cont_code}")
        if cont_name == "Antarctica":
            continue

        cont_mask = mask & (continent_arr == cont_code)
        if not cont_mask.any():
            out[cont_name] = 0.0
            continue

        row_counts = cont_mask.sum(axis=1).astype(np.float64)
        out[cont_name] = float(np.sum(row_counts * row_areas_km2))

    return out


def write_area_table_txt(path, records):
    with open(path, "w") as f:
        header = [
            "continent",
            "glwd_area_km2",
            "rhodes_glwd_intersection_area_km2",
            "simulated_classes123_area_km2",
            "intersection_over_glwd",
            "simulated_over_glwd",
            "simulated_over_intersection",
        ]
        f.write("\t".join(header) + "\n")
        for rec in records:
            f.write(
                f"{rec['continent']}\t"
                f"{rec['glwd_area_km2']:.3f}\t"
                f"{rec['rhodes_glwd_intersection_area_km2']:.3f}\t"
                f"{rec['simulated_classes123_area_km2']:.3f}\t"
                f"{rec['intersection_over_glwd']:.6f}\t"
                f"{rec['simulated_over_glwd']:.6f}\t"
                f"{rec['simulated_over_intersection']:.6f}\n"
            )


def write_area_table_csv(path, records):
    header = [
        "continent",
        "glwd_area_km2",
        "rhodes_glwd_intersection_area_km2",
        "simulated_classes123_area_km2",
        "intersection_over_glwd",
        "simulated_over_glwd",
        "simulated_over_intersection",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(records)


def get_area_scale_and_label(max_value):
    if max_value >= 1_000_000:
        return 1_000_000.0, "Wetland area (million km\u00b2)"
    if max_value >= 100_000:
        return 100_000.0, "Wetland area (hundred thousand km\u00b2)"
    return 1.0, "Wetland area (km\u00b2)"


def count_persistence_classes(mask, persistence_arr, valid_mask=None, class_codes=None):
    if class_codes is None:
        class_codes = COMBINED_CLASSES

    if valid_mask is None:
        valid_mask = np.ones(mask.shape, dtype=bool)

    counts = {}
    selected = mask & valid_mask

    for cls in class_codes:
        counts[int(cls)] = int(np.sum(selected & (persistence_arr == cls)))

    total = int(sum(counts.values()))
    return counts, total


# =============================================================================
# MAIN
# =============================================================================
def main():
    plt.rcParams.update({
        "font.size": 40,
        "figure.dpi": 1200,
        "savefig.dpi": 1200,
    })

    print("=" * 60)
    print("Compare intersection vs persistence, classes 1 to 3 combined, drylands only, 10 km tolerance")
    print("=" * 60)
    print(f"INTERSECTION_TIF = {INTERSECTION_TIF}")
    print(f"PERSISTENCE_TIF  = {PERSISTENCE_TIF}")
    print(f"DRYLAND_MASK_TIF = {DRYLAND_MASK_TIF}")
    print(f"GLWD_CLASS_TIF   = {GLWD_CLASS_TIF}")
    print(f"OUTDIR           = {OUTDIR}")
    print(f"OPEN_WATER_VALUE = {OPEN_WATER_VALUE}")
    print()

    # -------------------------------------------------------------------------
    # Load rasters
    # -------------------------------------------------------------------------
    print("[1/10] Loading rasters")
    intersection, _, inter_transform, inter_crs, _ = load_raster(INTERSECTION_TIF)
    persistence, _, pers_transform, pers_crs, pers_nodata = load_raster(PERSISTENCE_TIF)
    dryland, _, dry_transform, dry_crs, dry_nodata = load_raster(DRYLAND_MASK_TIF)
    glwd_class, _, glwd_transform, glwd_crs, glwd_nodata = load_raster(GLWD_CLASS_TIF)

    print(f"    Intersection shape: {intersection.shape}")
    print(f"    Persistence shape : {persistence.shape}")
    print(f"    Dryland shape     : {dryland.shape}")
    print(f"    GLWD class shape  : {glwd_class.shape}")
    print(f"    Unique intersection values: {np.unique(intersection)}")

    # -------------------------------------------------------------------------
    # Align rasters to intersection grid
    # -------------------------------------------------------------------------
    print("[2/10] Aligning persistence to intersection grid")
    persistence_aligned = align_to_grid(
        src_arr=persistence.astype(np.float32),
        src_transform=pers_transform,
        src_crs=pers_crs,
        src_nodata=float(pers_nodata) if pers_nodata is not None else float(PERSISTENCE_NODATA),
        dst_shape=intersection.shape,
        dst_transform=inter_transform,
        dst_crs=inter_crs,
        dst_nodata=float(PERSISTENCE_NODATA),
        resampling=Resampling.nearest,
    ).astype(np.int16)

    print("[3/10] Aligning dryland mask to intersection grid")
    dryland_aligned = align_to_grid(
        src_arr=dryland.astype(np.float32),
        src_transform=dry_transform,
        src_crs=dry_crs,
        src_nodata=float(dry_nodata) if dry_nodata is not None else np.nan,
        dst_shape=intersection.shape,
        dst_transform=inter_transform,
        dst_crs=inter_crs,
        dst_nodata=np.nan,
        resampling=Resampling.nearest,
    )

    dryland_safe_int = np.where(
        np.isfinite(dryland_aligned), dryland_aligned, -9999,
    ).astype(np.int16)
    dryland_mask = np.isfinite(dryland_aligned) & np.isin(
        dryland_safe_int, list(DRYLAND_CLASSES),
    )
    print(f"    Dryland pixels on intersection grid : {int(dryland_mask.sum()):,}")

    print("[4/10] Aligning GLWD class raster to intersection grid")
    glwd_aligned = align_to_grid(
        src_arr=glwd_class.astype(np.float32),
        src_transform=glwd_transform,
        src_crs=glwd_crs,
        src_nodata=float(glwd_nodata) if glwd_nodata is not None else float(GLWD_NODATA),
        dst_shape=intersection.shape,
        dst_transform=inter_transform,
        dst_crs=inter_crs,
        dst_nodata=float(GLWD_NODATA),
        resampling=Resampling.nearest,
    ).astype(np.int16)

    print("[5/10] Building continent raster on intersection grid")
    continent_raster, code_to_continent = build_continent_raster(
        dst_shape=intersection.shape,
        dst_transform=inter_transform,
    )

    # -------------------------------------------------------------------------
    # Restrict to dryland comparison bounding box
    # -------------------------------------------------------------------------
    print("[6/10] Building dryland comparison bounding box from reference and simulation")
    comparison_domain = (
        ((intersection == INTERSECTION_POSITIVE) & dryland_mask)
        | (np.isin(persistence_aligned, COMBINED_CLASSES) & dryland_mask)
    )
    r0, r1, c0, c1 = get_bbox_from_mask(comparison_domain, pad=PAD)

    print(f"    Bounding rows: {r0}:{r1}")
    print(f"    Bounding cols: {c0}:{c1}")
    print(f"    Cropped shape : {(int(r1 - r0), int(c1 - c0))}")

    intersection_crop = intersection[r0:r1, c0:c1]
    persistence_crop = persistence_aligned[r0:r1, c0:c1]
    dryland_crop = dryland_mask[r0:r1, c0:c1]
    glwd_crop = glwd_aligned[r0:r1, c0:c1]
    continent_crop = continent_raster[r0:r1, c0:c1]
    extent_crop = crop_extent(inter_transform, r0, r1, c0, c1)

    open_water_pixels = np.sum(intersection_crop == OPEN_WATER_VALUE)
    print(f"    Open water pixels in crop (excluded): {int(open_water_pixels):,}")

    # -------------------------------------------------------------------------
    # Prepare masks
    # -------------------------------------------------------------------------
    print("[7/10] Preparing comparison masks inside cropped dryland domain")
    valid = (
        np.isfinite(persistence_crop)
        & (persistence_crop != PERSISTENCE_NODATA)
        & dryland_crop
    )

    ref = (
        (intersection_crop == INTERSECTION_POSITIVE)
        & dryland_crop
        & valid
    ).astype(bool)

    pred_combined = (
        np.isin(persistence_crop, COMBINED_CLASSES)
        & dryland_crop
        & valid
    ).astype(bool)

    glwd_wetland = (
        (glwd_crop != GLWD_NODATA)
        & dryland_crop
        & valid
    ).astype(bool)

    print(f"    Valid comparison pixels         : {int(valid.sum()):,}")
    print(f"    Reference positive pixels       : {int(ref.sum()):,}")
    print(f"    Simulated combined positives    : {int(pred_combined.sum()):,}")
    print(f"    GLWD wetland pixels             : {int(glwd_wetland.sum()):,}")

    reference_positive_count = int(ref.sum())
    reference_summary = summarize_continent_glwd(
        ref, glwd_crop, continent_crop, glwd_nodata=GLWD_NODATA,
    )

    # -------------------------------------------------------------------------
    # Combined tolerance evaluation
    # -------------------------------------------------------------------------
    print("[8/10] Computing 10 km tolerance-based metrics for combined classes 1 to 3")
    metrics, missed_summary, tp_pred_mask, fp_mask, hit_ref_mask = evaluate_combined_tiled(
        pred=pred_combined,
        ref=ref,
        glwd_crop=glwd_crop,
        continent_crop=continent_crop,
        inter_transform=inter_transform,
        extent_crop=extent_crop,
        crop_row_offset=r0,
        crop_col_offset=c0,
        tolerance_km=TOLERANCE_KM,
        reference_positive_count=reference_positive_count,
        reference_summary=reference_summary,
    )

    print(f"    predicted_positive_pixels          : {metrics['predicted_positive_pixels']:,}")
    print(f"    tolerant_true_positive_pred_pixels : {metrics['tolerant_true_positive_pred_pixels']:,}")
    print(f"    tolerant_false_alarm_pixels        : {metrics['tolerant_false_alarm_pixels']:,}")
    print(f"    tolerant_reference_hits            : {metrics['tolerant_reference_hits']:,}")
    print(f"    tolerant_reference_misses          : {metrics['tolerant_reference_misses']:,}")
    print(f"    precision : {metrics['precision']}")
    print(f"    recall    : {metrics['recall']}")
    print(f"    f1        : {metrics['f1']}")

    for cont_code in sorted(missed_summary):
        cont_name = code_to_continent.get(cont_code, f"Continent_{cont_code}")
        if cont_name == "Antarctica":
            continue
        cont_total = sum(missed_summary[cont_code].values())
        print(f"    {cont_name}: {cont_total:,} missed pixels")

    tp_fp_plot, row_step, col_step = build_downsampled_plot_array(
        ref_full=ref,
        pred_full=pred_combined,
        tp_pred_full=tp_pred_mask,
        fp_full=fp_mask,
        hit_ref_full=hit_ref_mask,
    )

    # -------------------------------------------------------------------------
    # Class composition inside reference intersection
    # -------------------------------------------------------------------------
    print("[9/10] Computing persistence-class composition inside intersection, excluding class 0")
    inter_positive_classes = ref & np.isin(persistence_crop, COMBINED_CLASSES)

    class_counts = {
        "intersection_total_valid_pixels": int(ref.sum()),
        "intersection_pixels_class_1_to_3_only": int(inter_positive_classes.sum()),
        "class_1_episodic": 0,
        "class_2_seasonal": 0,
        "class_3_perennial": 0,
    }

    if inter_positive_classes.sum() > 0:
        vals, counts = np.unique(
            persistence_crop[inter_positive_classes], return_counts=True,
        )
        for v, c in zip(vals, counts):
            if int(v) == 1:
                class_counts["class_1_episodic"] = int(c)
            elif int(v) == 2:
                class_counts["class_2_seasonal"] = int(c)
            elif int(v) == 3:
                class_counts["class_3_perennial"] = int(c)

        total = int(inter_positive_classes.sum())
        class_counts["class_1_episodic_fraction"] = class_counts["class_1_episodic"] / total
        class_counts["class_2_seasonal_fraction"] = class_counts["class_2_seasonal"] / total
        class_counts["class_3_perennial_fraction"] = class_counts["class_3_perennial"] / total
    else:
        class_counts["class_1_episodic_fraction"] = np.nan
        class_counts["class_2_seasonal_fraction"] = np.nan
        class_counts["class_3_perennial_fraction"] = np.nan

    for k, v in class_counts.items():
        print(f"    {k}: {v}")

    # -------------------------------------------------------------------------
    # Area comparison by continent
    # -------------------------------------------------------------------------
    print("[10/10] Computing continent-wise wetland area summaries")
    crop_transform = window_transform(
        Window(c0, r0, c1 - c0, r1 - r0), inter_transform,
    )
    row_areas_km2 = get_row_area_km2(crop_transform, intersection_crop.shape[0])

    glwd_area = area_by_continent_km2(glwd_wetland, continent_crop, row_areas_km2, code_to_continent)
    ref_area = area_by_continent_km2(ref, continent_crop, row_areas_km2, code_to_continent)
    sim_area = area_by_continent_km2(pred_combined, continent_crop, row_areas_km2, code_to_continent)

    continents_out = sorted(
        set(glwd_area.keys()) | set(ref_area.keys()) | set(sim_area.keys()),
        key=lambda c: glwd_area.get(c, 0.0),
        reverse=True,
    )

    area_records = []
    for cont in continents_out:
        g = float(glwd_area.get(cont, 0.0))
        r = float(ref_area.get(cont, 0.0))
        s = float(sim_area.get(cont, 0.0))

        area_records.append({
            "continent": cont,
            "glwd_area_km2": g,
            "rhodes_glwd_intersection_area_km2": r,
            "simulated_classes123_area_km2": s,
            "intersection_over_glwd": (r / g) if g > 0 else np.nan,
            "simulated_over_glwd": (s / g) if g > 0 else np.nan,
            "simulated_over_intersection": (s / r) if r > 0 else np.nan,
        })

    # -------------------------------------------------------------------------
    # Write outputs
    # -------------------------------------------------------------------------
    metrics_txt = OUTDIR / "comparison_metrics_classes123_combined_drylands_5km.txt"
    composition_txt = OUTDIR / "intersection_class_composition_positive_only_drylands.txt"
    missed_glwd_txt = OUTDIR / "missed_reference_glwd_classes_by_continent_classes123_combined.txt"

    tp_fp_png = OUTDIR / "tp_fp_map_classes123_combined_drylands_5km_robinson.png"
    tp_fp_pdf = OUTDIR / "tp_fp_map_classes123_combined_drylands_5km_robinson.pdf"

    area_txt = OUTDIR / "wetland_area_by_continent_drylands_classes123_combined.txt"
    area_csv = OUTDIR / "wetland_area_by_continent_drylands_classes123_combined.csv"

    area_png = OUTDIR / "wetland_area_by_continent_drylands_classes123_combined_barplot_goodlegend.png"
    area_pdf = OUTDIR / "wetland_area_by_continent_drylands_classes123_combined_barplot.pdf"

    combined_png = OUTDIR / "combined_map_barplot_drylands_classes123.png"
    combined_pdf = OUTDIR / "combined_map_barplot_drylands_classes123.pdf"

    write_metrics_txt(metrics_txt, metrics)
    write_dict_txt(composition_txt, class_counts)
    write_missed_glwd_summary(
        missed_glwd_txt,
        missed_summary,
        code_to_continent=code_to_continent,
        glwd_class_names=GLWD_CLASS_NAMES,
    )

    plot_tp_fp_downsampled([tp_fp_png, tp_fp_pdf], tp_fp_plot, extent_crop, row_step, col_step)
    plot_area_barplot([area_png, area_pdf], area_records)
    plot_combined_figure([combined_png, combined_pdf], tp_fp_plot, extent_crop, area_records)

    write_area_table_txt(area_txt, area_records)
    write_area_table_csv(area_csv, area_records)

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    del persistence, persistence_aligned
    del dryland, dryland_aligned, dryland_safe_int
    del glwd_class, glwd_aligned
    del continent_raster
    del intersection
    del ref, pred_combined, glwd_wetland
    del tp_pred_mask, fp_mask, hit_ref_mask
    del tp_fp_plot

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print(f"    wrote: {metrics_txt}")
    print(f"    wrote: {composition_txt}")
    print(f"    wrote: {missed_glwd_txt}")
    print(f"    wrote: {tp_fp_png}")
    print(f"    wrote: {tp_fp_pdf}")
    print(f"    wrote: {area_txt}")
    print(f"    wrote: {area_csv}")
    print(f"    wrote: {area_png}")
    print(f"    wrote: {area_pdf}")
    print(f"    wrote: {combined_png}")
    print(f"    wrote: {combined_pdf}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()