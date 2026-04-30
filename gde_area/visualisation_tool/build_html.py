"""
build_html.py
Reads parquet files and shapefiles, computes climatology stats,
and writes a single self-contained index.html with everything embedded.

Usage (on your laptop after copying files from Snellius):
    python build_html.py

Edit the PATHS block below to point to your local copies.
"""

import json
import math
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.validation import make_valid

warnings.filterwarnings("ignore")

# ── PATHS ─────────────────────────────────────────────────────
# Snellius source paths (for reference):
#   biome shp:    /gpfs/work2/0/prjs1578/futurewetgde/shapefiles/biomes_new/biomes/wwf_terr_ecos.shp
#   country shp:  /gpfs/scratch1/shared/otoo0001/data/shapefiles/WB_countries_Admin0_10m/World_Countries_Generalized.shp
#   biome parq:   /gpfs/scratch1/shared/otoo0001/paper_2/output_gde/gde_area_by_biome_mask_1_14monthly.parquet
#   country parq: /gpfs/scratch1/shared/otoo0001/paper_2/output_gde/gde_area_by_country_monthly.parquet
#
# Copy to laptop first:
#   scp -r otoo0001@snellius.surf.nl:/gpfs/work2/0/prjs1578/futurewetgde/shapefiles/biomes_new/biomes/ ./shapefiles/
#   scp -r otoo0001@snellius.surf.nl:/gpfs/scratch1/shared/otoo0001/data/shapefiles/WB_countries_Admin0_10m/ ./shapefiles/
#   scp otoo0001@snellius.surf.nl:/gpfs/scratch1/shared/otoo0001/paper_2/output_gde/gde_area_by_biome_mask_1_14monthly.parquet ./parquets/
#   scp otoo0001@snellius.surf.nl:/gpfs/scratch1/shared/otoo0001/paper_2/output_gde/gde_area_by_country_monthly.parquet ./parquets/

BIOME_SHP          = Path("/gpfs/work2/0/prjs1578/futurewetgde/shapefiles/biomes_new/biomes/wwf_terr_ecos.shp")
COUNTRY_SHP        = Path("/gpfs/scratch1/shared/otoo0001/data/shapefiles/WB_countries_Admin0_10m/World_Countries_Generalized.shp")
BIOME_PARQ         = Path("/scratch-shared/otoo0001/paper2_revisions/output_area/gde_area_by_biome_climatology.parquet")
BIOME_REALM_PARQ   = Path("/scratch-shared/otoo0001/paper2_revisions/output_area/gde_area_by_biome_realm_climatology.parquet")
BIOME_COUNTRY_PARQ = Path("/scratch-shared/otoo0001/paper2_revisions/output_area/gde_area_by_biome_country_climatology.parquet")
OUTPUT_HTML        = Path("wetgde_explorer.html")
# ─────────────────────────────────────────────────────────────

BIOME_NAMES = {
    1:"Trop. Moist Broadleaf Forests",   2:"Trop. Dry Broadleaf Forests",
    3:"Trop. Coniferous Forests",         4:"Temp. Broadleaf & Mixed Forests",
    5:"Temp. Coniferous Forests",          6:"Boreal Forests / Taiga",
    7:"Trop. Grasslands & Savannas",      8:"Temp. Grasslands & Savannas",
    9:"Flooded Grasslands & Savannas",   10:"Montane Grasslands & Shrublands",
    12:"Mediterranean Forests",
    13:"Deserts & Xeric Shrublands",      14:"Mangroves",
}
REALM_NAMES = {
    "AA":"Australasia","AN":"Antarctic","AT":"Afrotropics",
    "IM":"Indomalaya","NA":"Nearctic","NT":"Neotropics",
    "PA":"Palearctic",
}
MONTH_ABBR = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ── Helpers ───────────────────────────────────────────────────
def clean_gdf(gdf, simplify=0.05):
    gdf = gdf.to_crs("EPSG:4326")
    gdf["geometry"] = gdf["geometry"].apply(lambda g: make_valid(g) if g is not None else g)
    gdf = gdf[gdf["geometry"].notna() & ~gdf["geometry"].is_empty].copy()
    if simplify:
        gdf["geometry"] = gdf["geometry"].simplify(simplify, preserve_topology=True)
    return gdf

def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"None of {candidates} in {list(df.columns)}")

def complete_months(df, group_cols, value_col="area_km2"):
    months   = pd.DataFrame({"month": range(1, 13)})
    groups   = df[group_cols].drop_duplicates()
    template = groups.merge(months, how="cross")
    df = template.merge(df, on=group_cols + ["month"], how="left")
    df[value_col] = df[value_col].fillna(0.0)
    return df

def compute_stats(df, group_cols, value_col="area_km2"):
    df    = complete_months(df, group_cols, value_col)
    stats = (df.groupby(group_cols)[value_col]
               .agg(annual_mean_km2="mean", annual_min_km2="min", annual_max_km2="max")
               .reset_index())
    stats["seasonal_range_km2"] = stats["annual_max_km2"] - stats["annual_min_km2"]
    stats["seasonal_range_pct"] = np.where(
        stats["annual_mean_km2"] > 0,
        100.0 * stats["seasonal_range_km2"] / stats["annual_mean_km2"], 0.0)
    std = df.groupby(group_cols)[value_col].std().reset_index()
    std.columns = group_cols + ["_std"]
    stats = stats.merge(std, on=group_cols, how="left")
    stats["cv"] = np.where(stats["annual_mean_km2"] > 0, stats["_std"] / stats["annual_mean_km2"], 0.0)
    stats.drop(columns=["_std"], inplace=True)
    peak   = df.loc[df.groupby(group_cols)[value_col].idxmax(), group_cols + ["month"]].rename(columns={"month":"peak_month"})
    trough = df.loc[df.groupby(group_cols)[value_col].idxmin(), group_cols + ["month"]].rename(columns={"month":"trough_month"})
    stats  = stats.merge(peak, on=group_cols, how="left").merge(trough, on=group_cols, how="left")
    monthly = df.pivot_table(index=group_cols, columns="month", values=value_col, aggfunc="mean").reset_index()
    monthly.columns = group_cols + [f"m{int(c):02d}_km2" for c in monthly.columns[len(group_cols):]]
    stats = stats.merge(monthly, on=group_cols, how="left")
    for m in range(1, 13):
        col = f"m{m:02d}_km2"
        stats[f"m{m:02d}_anom_km2"] = stats[col] - stats["annual_mean_km2"]
        stats[f"m{m:02d}_anom_pct"] = np.where(
            stats["annual_mean_km2"] > 0,
            100.0 * stats[f"m{m:02d}_anom_km2"] / stats["annual_mean_km2"], 0.0)
    return stats

def safe_json(obj):
    """Recursively replace NaN/Inf with None."""
    if isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_json(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if (math.isnan(float(obj)) or math.isinf(float(obj))) else round(float(obj), 4)
    return obj

def read_parq(path, biome_cands, extra=None):
    df = pd.read_parquet(path)
    df.columns = [c.strip() for c in df.columns]  # keep original case for ISO
    df_lower = {c.lower(): c for c in df.columns}

    def fc(cands):
        for c in cands:
            if c in df.columns: return c
            if c.lower() in df_lower: return df_lower[c.lower()]
        raise ValueError(f"None of {cands} in {list(df.columns)}")

    bc = fc(biome_cands)
    ac = fc(["area_km2","area","gde_area_km2"])

    # handle time column: extract month
    if "time" in df.columns:
        df["month"] = pd.to_datetime(df["time"]).dt.month
        mc = "month"
    else:
        mc = fc(["month"])

    rename = {bc:"biome_code", mc:"month", ac:"area_km2"}
    if extra:
        for target, cands in extra.items():
            rename[fc(cands)] = target
    df = df.rename(columns=rename)
    keep = ["biome_code","month","area_km2"] + list(extra.keys() if extra else [])
    df = df[[c for c in keep if c in df.columns]].copy()
    df["biome_code"] = df["biome_code"].astype(int)
    # cell area file: cdo_grid_area_30sec_map_correct_lat.nc
    # variable: cell_area, units: m², mean ~546,622 m² (0.547 km²) per 30-arcsec cell
    # wetGDE is binary at 30-arcsec — correct resolution match, just convert m²→km²
    df["area_km2"] = df["area_km2"] / 1e6
    print(f"  area_km2 after m²→km²: median={df['area_km2'].median():.2f}  max={df['area_km2'].max():.2f} km²")
    return df

# ── Load & process ────────────────────────────────────────────
print("Loading shapefiles...")
biome_shp   = clean_gdf(gpd.read_file(BIOME_SHP))
country_shp = clean_gdf(gpd.read_file(COUNTRY_SHP))

bc_col = find_col(biome_shp, ["BIOME","biome","BIOME_NUM"])
rc_col = next((c for c in ["REALM","realm","WWF_REALM","REALM2"] if c in biome_shp.columns), None)
biome_shp["biome_code"] = biome_shp[bc_col].astype(int)
biome_shp["realm"]      = biome_shp[rc_col].astype(str).str.strip() if rc_col else "UNK"

# exclude Lake (98) and Rock & Ice (99)
EXCLUDE_BIOMES = {11, 98, 99}  # Tundra, Lake, Rock & Ice
biome_shp = biome_shp[~biome_shp["biome_code"].isin(EXCLUDE_BIOMES)].copy()
print(f"Excluded biomes {EXCLUDE_BIOMES}, remaining: {sorted(biome_shp['biome_code'].unique())}")
EXCLUDE_REALMS = {"OC"}
biome_shp = biome_shp[~biome_shp["realm"].isin(EXCLUDE_REALMS)].copy()
print(f"Excluded realms {EXCLUDE_REALMS}")

ic_col = find_col(country_shp, ["ISO_A3","ISO3","ISO_3","ISO","iso_a3"])
nc_col = find_col(country_shp, ["COUNTRY","NAME","NAME_0","Country_Name","Country"])
country_shp["ISO"]          = country_shp[ic_col].astype(str).str.upper().str.strip()
country_shp["country_name"] = country_shp[nc_col].astype(str).str.strip()

print("Dissolving geometries...")
gdf_biome = clean_gdf(
    biome_shp.dissolve(by="biome_code", as_index=False)[["biome_code","geometry"]])
gdf_realm = clean_gdf(
    biome_shp.dissolve(by=["biome_code","realm"], as_index=False)[["biome_code","realm","geometry"]])
bc_join = gpd.overlay(
    biome_shp[["biome_code","geometry"]],
    country_shp[["ISO","country_name","geometry"]],
    how="intersection")
gdf_country = clean_gdf(
    bc_join.dissolve(by=["biome_code","ISO"], as_index=False)[["biome_code","ISO","country_name","geometry"]])

print("Loading parquets & computing stats...")
df_b  = read_parq(BIOME_PARQ,  ["biome_id","biome","biome_code","biome_num"])
df_bc = read_parq(BIOME_COUNTRY_PARQ, ["biome_id","biome","biome_code"], {"ISO":["ISO","iso","iso3","iso_a3","country_iso"]})
df_b  = df_b[~df_b["biome_code"].isin(EXCLUDE_BIOMES)].copy()
df_bc = df_bc[~df_bc["biome_code"].isin(EXCLUDE_BIOMES)].copy()
if BIOME_REALM_PARQ.exists():
    df_br = read_parq(BIOME_REALM_PARQ, ["biome_id","biome","biome_code"], {"realm":["realm","wwf_realm"]})
    df_br = df_br[~df_br["biome_code"].isin(EXCLUDE_BIOMES)].copy()
    df_br = df_br[~df_br["realm"].isin(EXCLUDE_REALMS)].copy()
    has_realm = True
else:
    df_br = None
    has_realm = False
    print("Skipping biome x realm layer (parquet not found)")
if has_realm:
    df_br["realm"] = df_br["realm"].astype(str).str.strip()
df_bc["ISO"]   = df_bc["ISO"].astype(str).str.upper().str.strip()

stats_b  = compute_stats(df_b,  ["biome_code"])
stats_bc = compute_stats(df_bc, ["biome_code","ISO"])
stats_br = compute_stats(df_br, ["biome_code","realm"]) if has_realm else None

global_mean = float(stats_b["annual_mean_km2"].sum())
for s in ([stats_b, stats_br, stats_bc] if has_realm else [stats_b, stats_bc]):
    s["pct_of_global"] = np.where(global_mean > 0, 100.0 * s["annual_mean_km2"] / global_mean, 0.0)

bt = stats_b[["biome_code","annual_mean_km2"]].rename(columns={"annual_mean_km2":"_bt"})
for s in ([stats_br, stats_bc] if has_realm else [stats_bc]):
    s.drop(columns=[c for c in s.columns if c=="_bt"], inplace=True, errors="ignore")
    merged = s.merge(bt, on="biome_code", how="left")
    s["pct_of_biome"] = np.where(merged["_bt"]>0, 100.0*s["annual_mean_km2"]/merged["_bt"], 0.0)

ct = stats_bc.groupby("ISO")["annual_mean_km2"].sum().reset_index().rename(columns={"annual_mean_km2":"_ct"})
stats_bc = stats_bc.merge(ct, on="ISO", how="left")
stats_bc["pct_of_country"] = np.where(stats_bc["_ct"]>0, 100.0*stats_bc["annual_mean_km2"]/stats_bc["_ct"], 0.0)
stats_bc.drop(columns=["_ct"], inplace=True)

stats_b["biome_name"]   = stats_b["biome_code"].map(BIOME_NAMES).fillna("Unknown")
stats_bc["biome_name"]  = stats_bc["biome_code"].map(BIOME_NAMES).fillna("Unknown")
if has_realm:
    stats_br["biome_name"] = stats_br["biome_code"].map(BIOME_NAMES).fillna("Unknown")
    stats_br["realm_name"] = stats_br["realm"].map(REALM_NAMES).fillna(stats_br["realm"])
    stats_br["label"]      = stats_br["biome_name"] + " – " + stats_br["realm_name"]
iso_map = country_shp[["ISO","country_name"]].drop_duplicates().set_index("ISO")["country_name"].to_dict()
stats_bc["country_name"] = stats_bc["ISO"].map(iso_map).fillna(stats_bc["ISO"])
stats_bc["label"]        = stats_bc["biome_name"] + " \u2013 " + stats_bc["country_name"]
stats_b["label"]         = stats_b["biome_name"]

# ── Risk index ────────────────────────────────────────────────
# Components (all 0-1 normalised, higher = more at risk / more important):
#   instability  = cv / max(cv)             (high cv = unstable)
#   swing        = seasonal_range_pct / 100 clipped to 1  (large swings)
#   dry_stress   = 1 - (annual_min / annual_mean)  clipped 0-1  (low dry-season floor)
#   value        = pct_of_global / max(pct_of_global)  (ecological importance)
# risk = 0.3*instability + 0.25*swing + 0.25*dry_stress + 0.20*value

def add_risk(df):
    eps = 1e-9
    cv_max   = df["cv"].replace([np.inf,-np.inf],np.nan).max() or eps
    pct_max  = df["pct_of_global"].replace([np.inf,-np.inf],np.nan).max() or eps
    instab   = (df["cv"].clip(0) / cv_max).clip(0,1)
    swing    = (df["seasonal_range_pct"].clip(0) / 100).clip(0,1)
    dry      = (1 - (df["annual_min_km2"] / df["annual_mean_km2"].replace(0,eps))).clip(0,1)
    value    = (df["pct_of_global"].clip(0) / pct_max).clip(0,1)
    df["risk_index"] = (0.30*instab + 0.25*swing + 0.25*dry + 0.20*value).round(4)
    # risk tier label
    def tier(v):
        if v >= 0.70: return "Critical"
        if v >= 0.50: return "High"
        if v >= 0.30: return "Moderate"
        return "Low"
    df["risk_tier"] = df["risk_index"].apply(tier)
    return df

stats_b  = add_risk(stats_b)
stats_bc = add_risk(stats_bc)
if has_realm and stats_br is not None:
    stats_br = add_risk(stats_br)

print("Building GeoJSON payloads...")
def make_unit_id(row, join_cols):
    parts = []
    for c in join_cols:
        parts.append(str(row[c]) if c in row.index else "x")
    return "__".join(parts)

def make_geojson(gdf, stats, join_cols, extra_props):
    merged = gdf.merge(stats, on=join_cols, how="left")
    merged = merged[merged["geometry"].notna() & ~merged["geometry"].is_empty].copy()
    # for unmatched rows: fill label/name strings, leave metrics as null (rendered as no-data)
    for col in ["label","biome_name","realm_name","country_name"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna("No data")
    # assign stable unit_id for JS selection
    merged["unit_id"] = merged.apply(lambda r: make_unit_id(r, join_cols), axis=1)
    features = []
    for _, row in merged.iterrows():
        cols = list(extra_props) + ["unit_id"]
        props = {c: row[c] for c in cols if c in merged.columns}
        props = safe_json(props)
        features.append({
            "type": "Feature",
            "geometry": row["geometry"].__geo_interface__,
            "properties": props,
        })
    return {"type":"FeatureCollection","features":features}

STAT_COLS = (
    ["biome_code","biome_name","label",
     "annual_mean_km2","annual_min_km2","annual_max_km2",
     "seasonal_range_km2","seasonal_range_pct","cv",
     "peak_month","trough_month","pct_of_global",
     "risk_index","risk_tier"]
    + [f"m{m:02d}_km2" for m in range(1,13)]
    + [f"m{m:02d}_anom_km2" for m in range(1,13)]
    + [f"m{m:02d}_anom_pct" for m in range(1,13)]
)

geojson_b  = make_geojson(gdf_biome,   stats_b,  ["biome_code"],         STAT_COLS)
geojson_br = make_geojson(gdf_realm,   stats_br, ["biome_code","realm"],  STAT_COLS + ["realm","realm_name","pct_of_biome"]) if has_realm else {"type":"FeatureCollection","features":[]}
geojson_bc = make_geojson(gdf_country, stats_bc, ["biome_code","ISO"],    STAT_COLS + ["ISO","country_name","pct_of_biome","pct_of_country"])

def js(obj):
    return json.dumps(safe_json(obj), separators=(",",":"))

print("Writing HTML...")

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Groundwater Dependent Wetlands — Climatology Explorer</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.29.1/plotly.min.js"></script>

<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Fraunces:opsz,wght@9..144,300;9..144,600&display=swap" rel="stylesheet"/>
<style>
:root{{
  /* base */
  --bg:#0b0f14;--panel:#111820;--panel2:#161e28;--ctrl:#1c2a38;
  --border:rgba(60,100,140,.20);--borderact:rgba(80,160,210,.55);
  --txt:#cfe0ee;--txt2:#6fa0be;--muted:#3d6070;
  --mono:'DM Mono',monospace;--serif:'Fraunces',Georgia,serif;
  /* metric accent colours — updated by JS on metric change */
  --acc:#3db8d8;       /* area = teal */
  --acc-glow:rgba(61,184,216,.20);
  /* named accents always available */
  --col-area:#3db8d8;
  --col-range:#e09040;
  --col-cv:#9060d0;
  --col-pct:#38c870;
  --col-anom-pos:#3aad82;
  --col-anom-neg:#d64c3c;
  --col-compare:#e8a45b;
}}
*{{box-sizing:border-box;margin:0;padding:0}}
html,body{{height:100%;background:var(--bg);color:var(--txt);font-family:var(--mono);font-size:13px;line-height:1.5}}

/* ── Header ── */
header{{height:58px;background:var(--panel);border-bottom:1px solid var(--border);
  display:flex;align-items:center;padding:0 20px;gap:14px;position:sticky;top:0;z-index:1000;
  box-shadow:0 2px 12px rgba(0,0,0,.4)}}
.badge{{background:var(--acc);color:#fff;font-size:9px;letter-spacing:.14em;
  text-transform:uppercase;padding:3px 8px;border-radius:3px;flex-shrink:0}}
header h1{{font-family:var(--serif);font-size:20px;font-weight:300;letter-spacing:.01em}}
.gsv{{font-family:var(--serif);font-size:18px;font-weight:600;color:var(--acc);
  transition:color .3s}}
.gsl{{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.09em}}
.gsu{{font-size:10px;color:var(--txt2)}}
.hright{{margin-left:auto;display:flex;align-items:baseline;gap:6px;
  background:var(--ctrl);border:1px solid var(--border);border-radius:5px;padding:5px 13px;
  transition:border-color .3s}}

/* ── Top bar ── */
.topbar{{height:50px;background:var(--panel2);border-bottom:1px solid var(--border);
  display:flex;align-items:center;gap:18px;padding:0 18px;
  position:sticky;top:58px;z-index:900;flex-wrap:wrap}}
.cg{{display:flex;align-items:center;gap:8px}}
.cg label{{font-size:10px;color:var(--txt2);text-transform:uppercase;letter-spacing:.07em;white-space:nowrap}}

/* ── Form controls ── */
select{{background:var(--ctrl);color:var(--txt);border:1px solid var(--border);
  border-radius:4px;padding:5px 9px;font-family:var(--mono);font-size:12px;
  cursor:pointer;outline:none;transition:border-color .15s}}
select:hover,select:focus{{border-color:var(--borderact)}}
select:disabled{{opacity:.4;cursor:not-allowed;pointer-events:none}}
input[type=range]{{-webkit-appearance:none;width:140px;height:3px;
  background:var(--ctrl);border-radius:2px;outline:none;cursor:pointer}}
input[type=range]::-webkit-slider-thumb{{-webkit-appearance:none;width:14px;height:14px;
  border-radius:50%;background:var(--acc);cursor:pointer;box-shadow:0 0 6px var(--acc-glow);
  transition:background .3s}}
button{{background:transparent;color:var(--txt2);border:1px solid var(--border);
  border-radius:4px;padding:5px 11px;font-family:var(--mono);font-size:12px;
  cursor:pointer;transition:color .15s,border-color .15s,background .15s}}
button:hover{{color:var(--acc);border-color:var(--borderact);background:rgba(61,184,216,.06)}}
button.active{{color:var(--acc);border-color:var(--acc);background:rgba(61,184,216,.10)}}

/* ── Layout ── */
.layout{{display:grid;grid-template-columns:224px 1fr 296px;
  height:calc(100vh - 58px - 50px);overflow:hidden}}
.lpanel{{background:var(--panel);border-right:1px solid var(--border);overflow-y:auto;padding:13px}}
.rpanel{{background:var(--panel);border-left:1px solid var(--border);overflow-y:auto}}

/* ── Left panel ── */
.ph{{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.12em;
  margin-bottom:7px;margin-top:14px;padding-bottom:4px;border-bottom:1px solid var(--border)}}
.ph:first-child{{margin-top:0}}
.fg label{{display:block;font-size:11px;color:var(--txt2);margin-bottom:3px}}
.fg select[multiple]{{width:100%;font-size:11px;padding:3px;border-radius:4px}}
.fg select[multiple] option:checked{{background:var(--acc);color:#fff}}
.fg.off label{{color:var(--muted)}}
.fg.off select{{opacity:.28;pointer-events:none}}
.fg{{margin-bottom:10px}}

/* ── Map ── */
#map{{width:100%;height:100%}}
.leaflet-tile{{filter:brightness(.52) saturate(.35) hue-rotate(5deg)}}
.leaflet-container{{background:#080e14}}
.leaflet-popup-content-wrapper{{background:var(--panel);color:var(--txt);
  border:1px solid var(--borderact);border-radius:6px;
  font-family:var(--mono);font-size:12px;box-shadow:0 4px 20px rgba(0,0,0,.5)}}
.leaflet-popup-tip{{background:var(--panel)}}
.leaflet-popup-content{{margin:10px 14px;line-height:1.6}}

/* ── Legend ── */
.legend{{position:absolute;bottom:26px;right:10px;background:rgba(11,15,20,.92);
  border:1px solid var(--border);border-radius:6px;padding:10px 13px;
  min-width:148px;z-index:800;backdrop-filter:blur(4px)}}
.lt{{font-size:10px;color:var(--txt2);letter-spacing:.06em;margin-bottom:6px;line-height:1.3}}
.lg{{height:10px;border-radius:3px;margin-bottom:4px}}
.ll{{display:flex;justify-content:space-between;font-size:10px;color:var(--muted)}}

/* ── Right panel sections ── */
.csec{{padding:11px 13px;border-bottom:1px solid var(--border)}}
.hint{{color:var(--muted);font-style:italic;font-size:11px}}
.slabel{{font-family:var(--serif);font-size:14px;font-weight:600;
  margin-bottom:6px;line-height:1.3;color:var(--txt)}}
.srow{{display:flex;justify-content:space-between;align-items:baseline;
  padding:3px 0;border-bottom:1px solid rgba(255,255,255,.035)}}
.srow:last-child{{border-bottom:none}}
.sk{{color:var(--muted);font-size:11px}}
.sv{{font-weight:500;font-size:12px}}

/* ── Summary cards ── */
.sgrid{{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:8px}}
.sc{{background:var(--ctrl);border:1px solid var(--border);border-radius:5px;
  padding:8px 10px;transition:border-color .2s}}
.sc:hover{{border-color:var(--borderact)}}
.scl{{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.09em;margin-bottom:3px}}
.scv{{font-family:var(--serif);font-size:15px;font-weight:600;color:var(--acc);
  transition:color .3s}}
.scu{{font-size:10px;color:var(--txt2);margin-left:2px}}

/* ── Chart sections ── */
.cm{{font-size:10px;color:var(--muted);margin-left:4px;font-weight:400}}
.chart{{width:100%;height:178px}}
#c-ranking{{height:215px}}

/* ── Metric colour indicator strip under topbar ── */
.metric-strip{{height:2px;background:var(--acc);transition:background .4s;
  position:sticky;top:108px;z-index:899}}

/* ── Insight panel ── */
.insight-text{{font-size:11px;line-height:1.75;color:var(--txt);}}
.insight-text em{{color:var(--txt2);font-style:normal;}}
.insight-highlight{{color:var(--acc);font-weight:500;}}
.insight-warn{{color:#fd8d3c;font-weight:500;}}
.insight-good{{color:#74c476;font-weight:500;}}
/* biome badge */
.biome-card{{background:var(--ctrl);border-left:3px solid var(--acc);
  border-radius:0 4px 4px 0;padding:7px 10px;margin-bottom:8px;font-size:11px;}}
.biome-card-name{{font-family:var(--serif);font-size:13px;font-weight:600;
  color:var(--txt);margin-bottom:3px;}}
.biome-card-desc{{color:var(--txt2);line-height:1.5;}}
/* comparison bar */
.cmp-bar-wrap{{margin:4px 0;}}
.cmp-bar-label{{display:flex;justify-content:space-between;font-size:10px;
  color:var(--txt2);margin-bottom:2px;}}
.cmp-bar-track{{height:6px;background:var(--ctrl);border-radius:3px;overflow:hidden;}}
.cmp-bar-fill{{height:100%;border-radius:3px;transition:width .4s;}}
/* risk legend */
.risk-legend-row{{display:flex;align-items:center;gap:8px;
  padding:4px 0;font-size:11px;}}
.risk-dot{{width:10px;height:10px;border-radius:50%;flex-shrink:0;}}
/* ── Overlay legend ── */
.overlay-legend{{position:absolute;bottom:110px;right:10px;background:rgba(11,15,20,.92);
  border:1px solid var(--border);border-radius:6px;padding:9px 12px;
  min-width:140px;z-index:800;display:none}}
.overlay-legend.visible{{display:block}}
.ol-title{{font-size:10px;color:var(--txt2);margin-bottom:6px;letter-spacing:.06em}}
.ol-row{{display:flex;align-items:center;gap:7px;font-size:10px;color:var(--txt2);margin:2px 0}}
.ol-swatch{{width:12px;height:12px;border-radius:2px;flex-shrink:0}}
.ol-anim-bar{{position:absolute;bottom:70px;right:10px;background:rgba(11,15,20,.88);
  border:1px solid var(--border);border-radius:5px;padding:6px 10px;
  z-index:800;display:none;align-items:center;gap:8px;font-size:11px;color:var(--txt2)}}
.ol-anim-bar.visible{{display:flex}}
/* ── Scrollbar ── */
::-webkit-scrollbar{{width:4px}}
::-webkit-scrollbar-track{{background:transparent}}
::-webkit-scrollbar-thumb{{background:var(--ctrl);border-radius:3px}}
::-webkit-scrollbar-thumb:hover{{background:var(--borderact)}}

@media(max-width:960px){{
  .layout{{grid-template-columns:1fr;grid-template-rows:auto 380px auto;height:auto;overflow:visible}}
  .lpanel,.rpanel{{border:none;border-bottom:1px solid var(--border)}}
  .topbar{{height:auto;padding:9px 14px;gap:10px}}
  .metric-strip{{display:none}}
}}
</style>
</head>
<body>
<header>
  <span class="badge">GDW</span>
  <h1>Groundwater Dependent Wetlands</h1>
  <div class="hright">
    <span class="gsl">Global Annual Mean</span>
    <span class="gsv" id="gsv">–</span>
    <span class="gsu">Mkm²</span>
  </div>
</header>

<div class="topbar">
  <div class="cg">
    <label>Layer</label>
    <select id="sel-layer">
      <option value="biome">Biome</option>
      <option value="biome_realm">Biome &times; Realm</option>
      <option value="biome_country">Biome &times; Country</option>
    </select>
  </div>
  <div class="cg">
    <label>Metric</label>
    <select id="sel-metric">
      <option value="annual_mean_km2">Annual Mean Area (km²)</option>
      <option value="annual_max_km2">Annual Max Area (km²)</option>
      <option value="annual_min_km2">Annual Min Area (km²)</option>
      <option value="seasonal_range_pct">Seasonal Range (%)</option>
      <option value="cv">Coeff. of Variation</option>
      <option value="pct_of_global">% of Global Total</option>
      <option value="risk_index">Hotspot Risk Index</option>
      <option value="monthly_area_km2">Monthly Area (km²) ▶</option>
    </select>
  </div>
  <div class="cg" id="month-grp" style="display:none">
    <label>Month: <span id="mlabel">Jan</span></label>
    <input type="range" id="month-sl" min="1" max="12" value="1"/>
    <button id="btn-play" title="Animate through months" style="padding:4px 10px;font-size:13px">&#9654;</button>
  </div>
  <div class="cg" id="overlay-grp">
    <label>Pixel Layer</label>
    <select id="sel-overlay">
      <option value="">None</option>
      <option value="persist_class">Persistence Class</option>
      <option value="wet_season_length">Wet Season Length</option>
      <option value="dry_stress">Dry Season Stress</option>
      <option value="risk_pixel">Pixel Risk</option>
      <option value="monthly">Monthly Presence ▶</option>
    </select>
  </div>
  <button id="btn-reset" style="margin-left:auto">&#8635; Reset</button>
</div>

<div class="metric-strip" id="metric-strip"></div>
<div class="layout">
  <!-- Left panel -->
  <div class="lpanel">
    <div class="ph">Filters</div>
    <div class="fg" id="fg-biome">
      <label>Biome</label>
      <select id="f-biome" multiple size="6"></select>
    </div>
    <div class="fg off" id="fg-realm">
      <label>Realm</label>
      <select id="f-realm" multiple size="5"></select>
    </div>
    <div class="fg off" id="fg-country">
      <label>Country</label>
      <select id="f-country" multiple size="6"></select>
    </div>
    <button id="btn-clear" style="width:100%;margin-top:4px">Clear Filters</button>
    <div class="ph" style="margin-top:14px">Country Search</div>
    <div style="display:flex;gap:5px;margin-bottom:8px">
      <input id="country-search" type="text" placeholder="Type country name..." autocomplete="off"
        style="flex:1;background:#1f2d3d;color:#d4e4f0;border:1px solid rgba(82,130,180,.18);
               border-radius:4px;padding:4px 7px;font-family:monospace;font-size:11px;outline:none"/>
      <button id="btn-country-search" title="Zoom to country" style="padding:4px 8px">&#9906;</button>
    </div>
    <div id="search-results" style="font-size:11px;color:#7fa8c4;max-height:80px;overflow-y:auto"></div>
    <div class="ph" style="margin-top:14px">Selection</div>
    <div id="status-box"><p class="hint">Click a polygon to inspect it.</p></div>
  </div>

  <!-- Map -->
  <div style="position:relative">
    <div id="map"></div>
    <div class="legend" id="legend"></div>
    <div class="overlay-legend" id="overlay-legend"></div>
    <div class="ol-anim-bar" id="ol-anim-bar">
      <button id="btn-ol-play" style="padding:3px 9px;font-size:13px">&#9654;</button>
      <span id="ol-month-label" style="min-width:28px;font-weight:600;color:var(--txt)">Jan</span>
      <input type="range" id="ol-month-sl" min="1" max="12" value="1" style="width:90px"/>
      <span id="ol-area-counter" style="font-size:10px;color:var(--txt2);white-space:nowrap"></span>
    </div>
  </div>

  <!-- Right panel -->
  <div class="rpanel">
    <div class="csec">
      <div class="ph" style="margin-top:0">Summary</div>
      <div id="summary-panel"><p class="hint">Select a feature.</p></div>
    </div>
    <div class="csec" id="sec-insight">
      <div class="ph" style="margin-top:0">Interpretation</div>
      <div id="insight-panel"><p class="hint">Select a feature to see interpretation.</p></div>
    </div>
    <div class="csec" id="sec-risk-legend" style="display:none">
      <div class="ph" style="margin-top:0">Risk Legend</div>
      <div id="risk-legend-panel"></div>
    </div>
    <div class="csec">
      <div class="ph" style="margin-top:0;display:flex;align-items:center;justify-content:space-between">
        <span>Seasonal Profile</span>
        <span style="display:flex;align-items:center;gap:6px">
          <span id="compare-label" style="font-size:10px;color:#7fa8c4;font-style:italic;max-width:110px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap"></span>
          <button id="btn-compare" title="Click a second polygon to compare" style="padding:2px 8px;font-size:10px">+ Compare</button>
          <button id="btn-clear-compare" style="padding:2px 8px;font-size:10px;display:none">&#x2715;</button>
        </span>
      </div>
      <div style="height:200px" id="c-seasonal"></div>
    </div>
    <div class="csec">
      <div class="ph" style="margin-top:0">Ranking <span class="cm" id="rank-meta"></span></div>
      <div id="c-ranking" style="width:100%;height:210px"></div>
    </div>
    <div class="csec">
      <div class="ph" style="margin-top:0">Composition <span class="cm">(annual mean area)</span></div>
      <div class="chart" id="c-comp"></div>
    </div>
  </div>
</div>

<script>
// ── Embedded data ─────────────────────────────────────────────
const LAYERS = {{
  biome:        {js(geojson_b)},
  biome_realm:  {js(geojson_br)},
  biome_country:{js(geojson_bc)},
}};
const GLOBAL_MEAN = {round(global_mean, 2)};
// Global reference stats for comparison panel (computed from biome layer)
const GLOBAL_STATS = {{
  mean_km2:    {round(float(stats_b["annual_mean_km2"].mean()), 2)},
  cv_mean:     {round(float(stats_b["cv"].mean()), 4)},
  cv_max:      {round(float(stats_b["cv"].max()), 4)},
  range_pct_mean: {round(float(stats_b["seasonal_range_pct"].mean()), 2)},
  risk_mean:   {round(float(stats_b["risk_index"].mean()), 4)},
}};
const MONTH_ABBR  = {json.dumps(MONTH_ABBR)};
const BIOME_NAMES = {js(BIOME_NAMES)};
const REALM_NAMES = {js(REALM_NAMES)};

// ── Biome context descriptions ───────────────────────────────
const BIOME_DESC = {{
  1:"Humid tropical forests with year-round rainfall. GDEs here are driven by shallow water tables in floodplains and riparian zones.",
  2:"Seasonally dry tropical forests. GDEs concentrate along river corridors and are highly sensitive to dry-season groundwater depth.",
  3:"Montane tropical conifer forests. GDEs often fed by orographic precipitation and lateral groundwater flows.",
  4:"Temperate mixed forests with moderate seasonality. GDEs include seeps, wet meadows and riparian woodlands.",
  5:"Cool temperate conifer forests. Snowmelt-driven groundwater recharge sustains GDEs through summer dry periods.",
  6:"Boreal forest with permafrost influence. GDEs are widespread but vulnerable to permafrost thaw altering drainage.",
  7:"Tropical savannas with pronounced wet-dry cycles. GDEs are key dry-season refugia for biodiversity.",
  8:"Temperate grasslands with continental climates. GDEs are geographically restricted and highly threatened by agriculture.",
  9:"Permanently or seasonally inundated grasslands. Among the highest GDE densities globally.",
  10:"High-altitude shrublands. GDEs fed by glacial and snowmelt water; highly vulnerable to cryosphere change.",
  12:"Mediterranean shrublands with summer drought. GDEs are critical dry-season refugia; strongly groundwater-dependent.",
  13:"Hyperarid deserts. GDEs are rare but disproportionately important as biodiversity hotspots and human water sources.",
  14:"Coastal mangrove forests. GDEs influenced by tidal groundwater and freshwater submarine discharge.",
}};

// ── State ─────────────────────────────────────────────────────
const S = {{
  layer:"biome", metric:"annual_mean_km2", month:1,
  fBiomes:[], fRealms:[], fCountries:[], selected:null,
  compareMode:false, compareP:null,
}};

// ── Colour scales per metric ─────────────────────────────────
// Area (YlGnBu-style dark): red/yellow=sparse → green → deep blue=dense
// This reads: more water = deeper blue, less = warm/yellow. Perceptually uniform.
const YlGnBu = ["#b10026","#e31a1c","#fc4e2a","#fd8d3c","#feb24c",
                 "#fed976","#ffffcc","#c7e9b4","#7fcdbb","#41b6c4",
                 "#1d91c0","#225ea8","#0c2c84"];
// Risk: green=low → yellow → red=critical (traffic-light, intuitive)
const RiskScale = ["#1a5c1a","#2e8b2e","#74c476","#bae4b3",
                   "#ffffb2","#fecc5c","#fd8d3c","#e31a1c","#800026"];
// Diverging anomaly: blue=below mean, white=at mean, red=above mean
const DivBWR = ["#2166ac","#4393c3","#92c5de","#d1e5f0",
                "#f5f5f5","#fddbc7","#f4a582","#d6604d","#b2182b"];
// Seasonal range / CV: white=stable → orange → dark brown=volatile
const VolScale = ["#fff7ec","#fee8c8","#fdd49e","#fdbb84",
                  "#fc8d59","#ef6548","#d7301f","#990000"];
// % contribution: white=none → teal → dark blue=dominant
const PctScale = ["#f7fcfd","#e0ecf4","#bfd3e6","#9ebcda",
                  "#8c96c6","#8c6bb1","#88419d","#6e016b"];

const SCALES = {{
  annual_mean_km2:    YlGnBu,
  annual_max_km2:     YlGnBu,
  annual_min_km2:     YlGnBu,
  monthly_area_km2:   YlGnBu,
  seasonal_range_km2: VolScale,
  seasonal_range_pct: VolScale,
  cv:                 VolScale,
  pct_of_global:      PctScale,
  pct_of_biome:       PctScale,
  pct_of_country:     PctScale,
  risk_index:         RiskScale,
  monthly_anomaly_km2:DivBWR,
  monthly_anomaly_pct:DivBWR,
}};
const SCALE_LABELS = {{
  annual_mean_km2:    "Annual Mean Area",
  annual_max_km2:     "Annual Max Area",
  annual_min_km2:     "Annual Min Area",
  seasonal_range_km2: "Seasonal Range",
  seasonal_range_pct: "Seasonal Range",
  cv:                 "Coefficient of Variation",
  pct_of_global:      "% of Global Total",
  pct_of_biome:       "% of Biome Total",
  pct_of_country:     "% of Country Total",
  risk_index:         "Hotspot Risk Index",
  monthly_area_km2:   "Monthly Area",
  monthly_anomaly_km2:"Monthly Anomaly",
  monthly_anomaly_pct:"Monthly Anomaly",
}};

function getStops(){{
  return SCALES[S.metric] || SCALES.annual_mean_km2;
}}

// ── Metric accent colours ─────────────────────────────────
const METRIC_ACCENT = {{
  annual_mean_km2:    "#1d91c0",
  annual_max_km2:     "#1d91c0",
  annual_min_km2:     "#1d91c0",
  monthly_area_km2:   "#1d91c0",
  seasonal_range_km2: "#ef6548",
  seasonal_range_pct: "#ef6548",
  cv:                 "#d7301f",
  pct_of_global:      "#8c6bb1",
  pct_of_biome:       "#8c6bb1",
  pct_of_country:     "#8c6bb1",
  risk_index:         "#e31a1c",
  monthly_anomaly_km2:"#4393c3",
  monthly_anomaly_pct:"#4393c3",
}};

function applyMetricAccent(){{
  const col = METRIC_ACCENT[S.metric] || "#3db8d8";
  const root = document.documentElement;
  root.style.setProperty("--acc", col);
  root.style.setProperty("--acc-glow", col+"33");
  const strip = document.getElementById("metric-strip");
  if(strip) strip.style.background = col;
}}

function lerp(a,b,t){{return a+(b-a)*t}}
function h2r(h){{const n=parseInt(h.slice(1),16);return[(n>>16)&255,(n>>8)&255,n&255]}}
function r2h(r,g,b){{return"#"+[r,g,b].map(v=>Math.round(v).toString(16).padStart(2,"0")).join("")}}
function cscale(v,mn,mx,div){{
  if(v==null||isNaN(v)) return"#2c2c2c";
  const stops=getStops();
  let t;
  if(div){{const e=Math.max(Math.abs(mn),Math.abs(mx))||1;t=v/e*0.5+0.5;}}
  else{{const r=mx-mn||1;t=(v-mn)/r;}}
  t=Math.max(0,Math.min(1,t));
  const si=Math.min(Math.floor(t*(stops.length-1)),stops.length-2);
  const lt=(t*(stops.length-1))-si;
  const [r1,g1,b1]=h2r(stops[si]);
  const [r2,g2,b2]=h2r(stops[si+1]);
  return r2h(lerp(r1,r2,lt),lerp(g1,g2,lt),lerp(b1,b2,lt));
}}

// ── Format ────────────────────────────────────────────────────
function fkm(v){{
  if(v==null||isNaN(v)) return"–";
  const n=Number(v);
  if(Math.abs(n)>=1e6) return(n/1e6).toFixed(2)+" M km²";
  if(Math.abs(n)>=1e3) return(n/1e3).toFixed(1)+" k km²";
  return n.toFixed(0)+" km²";
}}
function fv(v,unit){{
  if(v==null||isNaN(v)) return"–";
  const n=Number(v);
  if(unit==="km²"){{
    if(Math.abs(n)>=1e6) return(n/1e6).toFixed(2)+" M km²";
    if(Math.abs(n)>=1e3) return(n/1e3).toFixed(1)+" k km²";
    return n.toFixed(0)+" km²";
  }}
  return n.toFixed(unit=="%"?1:2)+(unit?" "+unit:"");
}}

// ── Metric helpers ────────────────────────────────────────────
const DIVERGING_METRICS = new Set(["monthly_anomaly_km2","monthly_anomaly_pct"]);
// metrics where lower = worse / more interesting (used for legend framing only)
const WARM_METRICS = new Set(["seasonal_range_km2","seasonal_range_pct","cv"]);
function getVal(p){{
  const m=S.metric;
  const mm=String(S.month).padStart(2,"00");
  if(m==="monthly_area_km2")    return p[`m${{mm}}_km2`]??null;
  if(m==="monthly_anomaly_km2") return p[`m${{mm}}_anom_km2`]??null;
  if(m==="monthly_anomaly_pct") return p[`m${{mm}}_anom_pct`]??null;
  return p[m]??null;
}}
function metricUnit(){{
  if(S.metric==="risk_index") return"";
  if(S.metric.endsWith("_pct")||S.metric.startsWith("pct_")||S.metric==="cv") return"%";
  if(S.metric.endsWith("_km2")) return"km²";
  return"";
}}
function isMonthlyMetric(){{
  return S.metric==="monthly_area_km2";
}}

// ── Map init ──────────────────────────────────────────────────
const map=L.map("map",{{center:[20,0],zoom:2}});
L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png",{{
  attribution:"&copy; OpenStreetMap contributors",maxZoom:18}}).addTo(map);
let gjLayer=null;

// ── Filter ────────────────────────────────────────────────────
function filtered(){{
  return LAYERS[S.layer].features.filter(f=>{{
    const p=f.properties;
    if(S.fBiomes.length&&!S.fBiomes.includes(String(p.biome_code))) return false;
    if(S.layer==="biome_realm"&&S.fRealms.length&&!S.fRealms.includes(p.realm)) return false;
    if(S.layer==="biome_country"&&S.fCountries.length&&!S.fCountries.includes(p.ISO)) return false;
    return true;
  }});
}}

// ── Build / refresh map layer ─────────────────────────────────
function buildMap(fitBounds){{
  const feats=filtered();
  const vals=feats.map(f=>getVal(f.properties)).filter(v=>v!=null&&!isNaN(v));
  const mn=vals.length?Math.min(...vals):0;
  const mx=vals.length?Math.max(...vals):1;
  const div=DIVERGING_METRICS.has(S.metric);
  if(gjLayer){{map.removeLayer(gjLayer);gjLayer=null;}}
  // if pixel overlay active, keep gjLayer hidden after build
  const pixelActive = document.getElementById("sel-overlay")?.value;
  gjLayer=L.geoJSON({{type:"FeatureCollection",features:feats}},{{
    style:f=>{{
      const p=f.properties;
      const v=getVal(p);
      const sel=(p.unit_id&&p.unit_id===S.selected)||(p.label&&p.label===S.selected);
      return{{fillColor:cscale(v,mn,mx,div),fillOpacity:sel?.85:.6,
              color:sel?"#5bc4e8":"rgba(82,130,180,.22)",weight:sel?2:.5}};
    }},
    onEachFeature:(f,l)=>{{
      const p=f.properties;
      l.on({{
        click:()=>selectFeat(p),
        mouseover:e=>{{
          const v=getVal(p);
          const u=metricUnit();
          const TIER_C={{Critical:"#e31a1c",High:"#fd8d3c",Moderate:"#fecc5c",Low:"#74c476"}};
          const riskHtml=p.risk_tier?`<br/><span style="color:${{TIER_C[p.risk_tier]||"#aaa"}}">&#9679; ${{p.risk_tier}} risk</span> <span style="color:#6fa0be">(${{Number(p.risk_index).toFixed(3)}})</span>`:"";
          l.bindPopup(`<strong>${{p.label||p.biome_name}}</strong>
            ${{p.country_name?`<br/><span style="color:#6fa0be">${{p.country_name}}${{p.ISO?" ("+p.ISO+")":""}}</span>`:""}}<br/>
            <span style="color:#6fa0be">${{SCALE_LABELS[S.metric]||S.metric.replace(/_/g," ")}}</span>: ${{fv(v,u)}}${{riskHtml}}`).openPopup();
        }},
      }});
    }},
  }});
  if(!pixelActive) gjLayer.addTo(map);
  updateLegend(mn,mx,div);
  if(fitBounds&&gjLayer){{
    try{{const b=gjLayer.getBounds();if(b.isValid())map.fitBounds(b,{{padding:[16,16]}});}}catch(_){{}}
  }}
}}

function refreshStyles(){{
  if(!gjLayer) return;
  const feats=filtered();
  const vals=feats.map(f=>getVal(f.properties)).filter(v=>v!=null&&!isNaN(v));
  const mn=vals.length?Math.min(...vals):0;
  const mx=vals.length?Math.max(...vals):1;
  const div=DIVERGING_METRICS.has(S.metric);
  gjLayer.eachLayer(l=>{{
    const p=l.feature.properties;
    const v=getVal(p);
    const sel=(p.unit_id&&p.unit_id===S.selected)||(p.label&&p.label===S.selected);
    l.setStyle({{fillColor:cscale(v,mn,mx,div),fillOpacity:sel?.85:.6,
                 color:sel?"#ffffff":"rgba(82,130,180,.22)",weight:sel?2.5:.5}});
  }});
  updateLegend(mn,mx,div);
}}

// ── Legend ────────────────────────────────────────────────────
function updateLegend(mn,mx,div){{
  const stops=getStops();
  const grad=stops.map((c,i)=>`${{c}} ${{Math.round(i/(stops.length-1)*100)}}%`).join(",");
  const u=metricUnit();
  const lo=div?"−"+fv(Math.max(Math.abs(mn),Math.abs(mx)),u):fv(mn,u);
  const hi=div?"+"+fv(Math.max(Math.abs(mn),Math.abs(mx)),u):fv(mx,u);
  let label=SCALE_LABELS[S.metric]||S.metric.replace(/_/g," ");
  if(isMonthlyMetric()) label+=` – ${{MONTH_ABBR[S.month-1]}}`;
  document.getElementById("legend").innerHTML=`
    <div class="lt">${{label}}${{u?" ("+u+")":""}}</div>
    <div class="lg" style="background:linear-gradient(to right,${{grad}})"></div>
    <div class="ll"><span>${{lo}}</span><span>${{hi}}</span></div>`;
  // show/hide risk legend panel
  const isRisk = S.metric==="risk_index";
  document.getElementById("sec-risk-legend").style.display = isRisk?"block":"none";
  if(isRisk) buildRiskLegend();
}}

function buildRiskLegend(){{
  const tiers=[
    {{label:"Critical",color:"#e31a1c",range:"≥ 0.70",desc:"High value + high instability"}},
    {{label:"High",    color:"#fd8d3c",range:"0.50–0.70",desc:"Significant seasonal stress"}},
    {{label:"Moderate",color:"#fecc5c",range:"0.30–0.50",desc:"Moderate vulnerability"}},
    {{label:"Low",     color:"#74c476",range:"< 0.30",  desc:"Stable, lower priority"}},
  ];
  document.getElementById("risk-legend-panel").innerHTML=tiers.map(t=>`
    <div class="risk-legend-row">
      <div class="risk-dot" style="background:${{t.color}}"></div>
      <div>
        <span style="color:${{t.color}};font-weight:500">${{t.label}}</span>
        <span style="color:var(--muted);margin-left:5px">${{t.range}}</span><br/>
        <span style="color:var(--txt2);font-size:10px">${{t.desc}}</span>
      </div>
    </div>`).join("");
}}

// ── Selection ─────────────────────────────────────────────────
function getSelectedProps(){{
  if(!S.selected) return null;
  const feat=LAYERS[S.layer].features.find(f=>f.properties.unit_id===S.selected);
  return feat?feat.properties:null;
}}

function selectFeat(p){{
  if(!p) return;
  const uid = p.unit_id || null;

  if(S.compareMode && S.selected){{
    S.compareP = p;
    S.compareMode = false;
    const compareName = p.label||p.biome_name||"";
    document.getElementById("compare-label").textContent = "vs. "+compareName;
    document.getElementById("btn-compare").textContent = "+ Compare";
    document.getElementById("btn-compare").classList.remove("active");
    document.getElementById("btn-clear-compare").style.display = "inline";
    const primary = getSelectedProps();
    refreshStyles();
    chartSeasonal(primary);
    // append compare note to insight panel
    const insightEl = document.getElementById("insight-panel");
    if(insightEl && primary){{
      const cmpNote = `<div style="margin-top:10px;padding:8px 10px;
        border-left:3px solid var(--col-compare);border-radius:0 4px 4px 0;
        background:var(--ctrl);font-size:11px;color:var(--txt2)">
        <span style="color:var(--col-compare);font-weight:500">Comparing with:</span>
        ${{compareName}}<br/>
        <span style="color:var(--muted);font-size:10px">
          Mean: ${{fkm(p.annual_mean_km2)}} &nbsp;|&nbsp;
          CV: ${{p.cv!=null?Number(p.cv).toFixed(2):"–"}} &nbsp;|&nbsp;
          Risk: <span style="color:${{{{Critical:"#e31a1c",High:"#fd8d3c",Moderate:"#fecc5c",Low:"#74c476"}}[p.risk_tier]||"#aaa"}}">${{p.risk_tier||"–"}}</span>
        </span>
      </div>`;
      // remove any existing compare note first
      const existing = insightEl.querySelector(".compare-note");
      if(existing) existing.remove();
      const div = document.createElement("div");
      div.className = "compare-note";
      div.innerHTML = cmpNote;
      insightEl.appendChild(div);
    }}
    return;
  }}

  S.selected = uid || p.label || p.biome_name;
  // new primary selection clears compare
  S.compareP = null;
  document.getElementById("compare-label").textContent = "";
  document.getElementById("btn-clear-compare").style.display = "none";
  document.getElementById("btn-compare").classList.remove("active");
  const cn=document.getElementById("insight-panel")?.querySelector(".compare-note");
  if(cn) cn.remove();
  refreshStyles();
  updateStatus(p);
  updateSummary(p);
  updateInsight(p);
  chartSeasonal(p);
  chartRanking();
  chartComp();
}}

function updateStatus(p){{
  if(!p){{
    document.getElementById("status-box").innerHTML='<p class="hint">Click a polygon to inspect it.</p>';
    document.getElementById("insight-panel").innerHTML='<p class="hint">Select a feature to see interpretation.</p>';
    return;
  }}
  const rows=[
    ["Biome",p.biome_name||"–"],
    p.realm_name?["Realm",p.realm_name]:null,
    p.country_name?["Country",(p.country_name||"")+(p.ISO?" ("+p.ISO+")":"")]:null,
  ].filter(Boolean);
  document.getElementById("status-box").innerHTML=`
    <div class="slabel">${{p.label||p.biome_name}}</div>
    ${{rows.map(([k,v])=>`<div class="srow"><span class="sk">${{k}}</span><span class="sv">${{v}}</span></div>`).join("")}}`;
}}

function updateSummary(p){{
  if(!p){{document.getElementById("summary-panel").innerHTML='<p class="hint">Select a feature.</p>';return;}}
  const cards=[
    ["Annual Mean",fkm(p.annual_mean_km2),""],
    ["Annual Max", fkm(p.annual_max_km2),""],
    ["Seas. Range",fkm(p.seasonal_range_km2),""],
    ["CV",         p.cv!=null?Number(p.cv).toFixed(3):"–",""],
    ["Peak",       p.peak_month?MONTH_ABBR[p.peak_month-1]:"–",""],
    ["Dry Month",  p.trough_month?MONTH_ABBR[p.trough_month-1]:"–",""],
    ["% Global",   p.pct_of_global!=null?Number(p.pct_of_global).toFixed(1):"–","%"],
    p.pct_of_biome!=null?["% Biome",Number(p.pct_of_biome).toFixed(1),"%"]:null,
    p.pct_of_country!=null?["% Country",Number(p.pct_of_country).toFixed(1),"%"]:null,
    p.risk_index!=null?["Risk Index",Number(p.risk_index).toFixed(3),""]:null,
    p.risk_tier?["Risk Tier",p.risk_tier,""]:null,
  ].filter(Boolean);
  const TIER_COLORS={{Critical:"#e31a1c",High:"#fd8d3c",Moderate:"#fecc5c",Low:"#74c476"}};
  document.getElementById("summary-panel").innerHTML=`<div class="sgrid">
    ${{cards.map(([l,v,u])=>{{
      const isTier=l==="Risk Tier";
      const col=isTier?(TIER_COLORS[v]||"var(--acc)"):"var(--acc)";
      return`<div class="sc"><div class="scl">${{l}}</div>
        <span class="scv" style="color:${{col}}">${{v}}</span><span class="scu">${{u}}</span></div>`;
    }}).join("")}}
  </div>`;
}}

// ── Insight panel ────────────────────────────────────────────
function cmpBar(label, val, globalVal, maxVal, color){{
  const pct = Math.min(100, Math.round((val/maxVal)*100));
  const gPct = Math.min(100, Math.round((globalVal/maxVal)*100));
  return `<div class="cmp-bar-wrap">
    <div class="cmp-bar-label"><span>${{label}}</span>
      <span>${{typeof val==="number"?val.toFixed(2):val}}</span></div>
    <div class="cmp-bar-track">
      <div class="cmp-bar-fill" style="width:${{pct}}%;background:${{color}}"></div>
    </div>
    <div style="font-size:9px;color:var(--muted);text-align:right;margin-top:1px">
      global avg ${{typeof globalVal==="number"?globalVal.toFixed(2):globalVal}}</div>
  </div>`;
}}

function updateInsight(p){{
  const el = document.getElementById("insight-panel");
  if(!p){{ el.innerHTML='<p class="hint">Select a feature to see interpretation.</p>'; return; }}

  const name  = p.label || p.biome_name || "This unit";
  const bcode = p.biome_code;
  const bmean = p.annual_mean_km2 || 0;
  const bmin  = p.annual_min_km2  || 0;
  const bmax  = p.annual_max_km2  || 0;
  const cv    = p.cv    || 0;
  const rng   = p.seasonal_range_pct || 0;
  const peak  = p.peak_month  ? MONTH_ABBR[p.peak_month-1]  : "unknown";
  const trough= p.trough_month? MONTH_ABBR[p.trough_month-1]:"unknown";
  const pctG  = p.pct_of_global || 0;
  const risk  = p.risk_index || 0;
  const tier  = p.risk_tier  || "Low";
  const TIER_C= {{Critical:"#e31a1c",High:"#fd8d3c",Moderate:"#fecc5c",Low:"#74c476"}};

  // dry stress
  const dryStress = bmean>0 ? 1-(bmin/bmean) : 0;
  const dryWord   = dryStress>0.7?"near-complete"
                  : dryStress>0.4?"substantial"
                  : dryStress>0.2?"moderate":"low";

  // stability
  const stabWord = cv<0.1?"very stable" : cv<0.25?"moderately stable"
                 : cv<0.5?"variable"    : "highly variable";

  // range word
  const rngWord = rng<20?"narrow" : rng<50?"moderate" : rng<100?"wide":"extreme";

  // significance
  const sigWord = pctG>5?"globally significant"
                : pctG>1?"regionally important"
                : pctG>0.1?"locally notable":"minor contributor";

  // biome card
  const bdesc = BIOME_DESC[bcode]||"";
  const biomeCard = bdesc?`<div class="biome-card">
    <div class="biome-card-name">${{BIOME_NAMES[bcode]||"Unknown biome"}}</div>
    <div class="biome-card-desc">${{bdesc}}</div>
  </div>`:"";

  // plain-English interpretation
  const interp = `<div class="insight-text">
    <span class="insight-highlight">${{name}}</span> has a
    <span class="insight-highlight">${{rngWord}}</span> seasonal cycle,
    peaking in <em>${{peak}}</em> and driest in <em>${{trough}}</em>.
    Wetland extent is <span class="insight-highlight">${{stabWord}}</span>
    across the year (CV ${{cv.toFixed(2)}}).
    ${{dryStress>0.2?`Dry-season stress is <span class="${{dryStress>0.5?"insight-warn":"insight-highlight"}}">${{dryWord}}</span>
    — the minimum monthly extent is ${{Math.round((1-dryStress)*100)}}% of the annual mean.`:
    "Dry-season extent remains close to the annual mean."}}
    This unit is <em>${{sigWord}}</em>, accounting for
    <span class="insight-highlight">${{pctG.toFixed(2)}}%</span> of global GDW extent.
    The composite risk classification is
    <span style="color:${{TIER_C[tier]}};font-weight:600">${{tier}}</span>
    (score ${{risk.toFixed(3)}}).
  </div>`;

  // comparison bars
  const GS = GLOBAL_STATS;
  const compSection = `
    <div class="ph" style="margin-top:10px">vs. Global Average</div>
    ${{cmpBar("CV (instability)", cv, GS.cv_mean, GS.cv_max, METRIC_ACCENT["cv"])}}
    ${{cmpBar("Seasonal range %", rng, GS.range_pct_mean, 200, METRIC_ACCENT["seasonal_range_pct"])}}
    ${{cmpBar("Risk index", risk, GS.risk_mean, 1, METRIC_ACCENT["risk_index"])}}
  `;

  el.innerHTML = biomeCard + interp + compSection;
}}

// ── Plotly theme ──────────────────────────────────────────────
const PT={{paper_bgcolor:"transparent",plot_bgcolor:"transparent",
  font:{{family:"DM Mono,monospace",size:10,color:"#6fa0be"}},
  margin:{{l:40,r:14,t:8,b:30}},autosize:true}};
const PC={{
  displayModeBar:"hover",
  modeBarButtonsToRemove:["zoom2d","pan2d","select2d","lasso2d","zoomIn2d","zoomOut2d","autoScale2d","resetScale2d","toggleSpikelines","hoverClosestCartesian","hoverCompareCartesian"],
  modeBarButtonsToAdd:["toImage"],
  toImageButtonOptions:{{format:"png",scale:2}},
  responsive:true
}};

// ── Charts ────────────────────────────────────────────────────
function buildSeasonalTraces(p,color,suffix){{
  if(!p) return [];
  const y   =Array.from({{length:12}},(_,i)=>p[`m${{String(i+1).padStart(2,"0")}}_km2`]??null);
  const anom=Array.from({{length:12}},(_,i)=>p[`m${{String(i+1).padStart(2,"0")}}_anom_km2`]??0);
  const mean=p.annual_mean_km2||0;
  const mn  =p.annual_min_km2||0;
  const mx  =p.annual_max_km2||0;
  const lbl =suffix?`${{(p.label||p.biome_name||"").slice(0,22)}} (${{suffix}})`:(p.label||p.biome_name||"").slice(0,28);
  // anomaly fill band (area between monthly value and mean)
  const yUpper=y.map((v,i)=>anom[i]>=0?v:mean);
  const yLower=y.map((v,i)=>anom[i]<0?v:mean);
  return [
    // min-max envelope
    {{x:[...MONTH_ABBR,...[...MONTH_ABBR].reverse()],
      y:[...Array(12).fill(mx),...Array(12).fill(mn)],
      fill:"toself",fillcolor:color.replace(")",",0.08)").replace("rgb","rgba"),
      line:{{color:"transparent"}},showlegend:false,hoverinfo:"skip",type:"scatter"}},
    // positive anomaly fill
    {{x:[...MONTH_ABBR,...[...MONTH_ABBR].reverse()],
      y:[...yUpper,...[...Array(12).fill(mean)].reverse()],
      fill:"toself",fillcolor:"rgba(58,173,130,0.25)",
      line:{{color:"transparent"}},showlegend:false,hoverinfo:"skip",type:"scatter"}},
    // negative anomaly fill
    {{x:[...MONTH_ABBR,...[...MONTH_ABBR].reverse()],
      y:[...yLower,...[...Array(12).fill(mean)].reverse()],
      fill:"toself",fillcolor:"rgba(214,76,60,0.25)",
      line:{{color:"transparent"}},showlegend:false,hoverinfo:"skip",type:"scatter"}},
    // annual mean dashed
    {{x:MONTH_ABBR,y:Array(12).fill(mean),type:"scatter",mode:"lines",
      name:`Mean${{suffix?" ("+suffix+")":""}}`,
      line:{{color:color,width:1.2,dash:"dot"}},hoverinfo:"skip"}},
    // main area line
    {{x:MONTH_ABBR,y,type:"scatter",mode:"lines+markers",name:lbl,
      line:{{color:color,width:2}},
      marker:{{size:5,color:color}},
      hovertemplate:`<b>%{{x}}</b><br>%{{y:,.1f}} km²<extra></extra>`}},
    // cv + peak annotation (invisible trace for legend entry)
    {{x:[MONTH_ABBR[( (p.peak_month||1)-1 )]],
      y:[y[(p.peak_month||1)-1]],
      type:"scatter",mode:"markers",
      marker:{{symbol:"star",size:10,color:color}},
      name:`Peak: ${{MONTH_ABBR[(p.peak_month||1)-1]}} | CV: ${{p.cv!=null?Number(p.cv).toFixed(2):"–"}}`,
      hovertemplate:`Peak: ${{MONTH_ABBR[(p.peak_month||1)-1]}}<br>CV: ${{p.cv!=null?Number(p.cv).toFixed(2):"–"}}<extra></extra>`}},
  ];
}}

function chartSeasonal(p){{
  const el=document.getElementById("c-seasonal");
  if(!p){{Plotly.react(el,[],{{...PT,title:{{text:"No selection",font:{{size:11,color:"#4a6a82"}}}}}},PC);return;}}
  const acc=METRIC_ACCENT[S.metric]||"#3db8d8";
  function hexToRgb(h){{const n=parseInt(h.slice(1),16);return`rgb(${{(n>>16)&255}},${{(n>>8)&255}},${{n&255}})`;}}
  const traces=[
    ...buildSeasonalTraces(p,hexToRgb(acc),S.compareP?"A":null),
    ...(S.compareP?buildSeasonalTraces(S.compareP,"rgb(232,164,91)","B"):[]),
  ];
  Plotly.react(el,traces,{{...PT,
    yaxis:{{title:{{text:"km²",font:{{size:10,color:"var(--txt2)"}}}},
      gridcolor:"rgba(60,100,140,.15)",zerolinecolor:"rgba(60,100,140,.25)",
      tickformat:".3s"}},
    xaxis:{{gridcolor:"rgba(82,130,180,.08)"}},
    legend:{{font:{{size:9}},bgcolor:"transparent",orientation:"h",y:-0.25}},
    height:200,margin:{{l:44,r:10,t:8,b:50}}}},PC);
}}

function chartRanking(){{
  const el=document.getElementById("c-ranking");
  const feats=filtered();
  if(!feats.length){{Plotly.react(el,[],{{...PT}},PC);return;}}
  const u=metricUnit();
  document.getElementById("rank-meta").textContent=`by ${{S.metric.replace(/_/g," ")}}${{u?" ("+u+")":""}}`;
  const pairs=feats
    .map(f=>{{const p=f.properties;
      return{{fullLabel:p.label||p.biome_name||"",
              shortLabel:(p.label||p.biome_name||"").slice(0,26),
              uid:p.unit_id||p.label||p.biome_name||"",
              val:getVal(p),props:p}};  }})
    .filter(d=>d.val!=null&&!isNaN(d.val))
    .sort((a,b)=>b.val-a.val).slice(0,20);
  if(!pairs.length){{Plotly.react(el,[],{{...PT}},PC);return;}}
  Plotly.react(el,[{{
    type:"bar",orientation:"h",
    x:pairs.map(d=>d.val),
    y:pairs.map(d=>d.shortLabel),
    customdata:pairs.map(d=>d.uid||d.fullLabel),
    marker:{{color:pairs.map(d=>d.uid===S.selected||d.shortLabel===S.selected?"#5bc4e8":"#2ea8d8"),opacity:.85}},
    text:pairs.map(d=>Number(d.val).toFixed(1)),
    textposition:"outside",textfont:{{size:9,color:"#7fa8c4"}},
    hovertemplate:"<b>%{{customdata}}</b><br>%{{x:.2f}} "+u+"<extra></extra>",
  }}],{{...PT,yaxis:{{autorange:"reversed",tickfont:{{size:9}},
    gridcolor:"rgba(82,130,180,.12)"}},
    xaxis:{{title:{{text:u,font:{{size:9}}}},gridcolor:"rgba(82,130,180,.12)"}},
    margin:{{l:120,r:44,t:8,b:28}},height:210}},PC);

  el.on("plotly_click",data=>{{
    const uid=data.points[0].customdata;
    const feat=filtered().find(f=>f.properties.unit_id===uid||(f.properties.label||f.properties.biome_name)===uid);
    if(feat) selectFeat(feat.properties);
  }});
}}

function chartComp(){{
  const el=document.getElementById("c-comp");
  const feats=filtered();
  if(!feats.length){{Plotly.react(el,[],{{...PT}},PC);return;}}
  const pairs=feats
    .map(f=>{{const p=f.properties;return{{label:(p.label||p.biome_name||"").slice(0,26),uid:p.unit_id||"",val:p.annual_mean_km2||0}};  }})
    .filter(d=>d.val>0).sort((a,b)=>b.val-a.val);
  const top=pairs.slice(0,8);
  const rest=pairs.slice(8).reduce((s,d)=>s+d.val,0);
  if(rest>0) top.push({{label:"Other",val:rest}});
  const COLORS=["#2ea8d8","#3aad82","#e8a45b","#9b6fc8","#d64c3c","#5bc4e8","#57c49b","#f0c070","#b88de0"];
  Plotly.react(el,[{{type:"pie",hole:.42,
    labels:top.map(d=>d.label),values:top.map(d=>d.val),
    marker:{{colors:COLORS.slice(0,top.length)}},
    textinfo:"percent",textfont:{{size:10,color:"#d4e4f0"}},
    pull:top.map(d=>(d.uid&&d.uid===S.selected)||d.label===S.selected?.06:0),
    hovertemplate:"<b>%{{label}}</b><br>%{{value:.1f}} km²<br>%{{percent}}<extra></extra>",
  }}],{{...PT,showlegend:false,margin:{{l:0,r:0,t:0,b:0}},height:175}},PC);
}}

// ── Filter helpers ────────────────────────────────────────────
function selVals(id){{return Array.from(document.getElementById(id).selectedOptions).map(o=>o.value);}}

function syncFilters(){{
  document.getElementById("fg-realm").className   ="fg"+(S.layer==="biome_realm"?"":" off");
  document.getElementById("fg-country").className ="fg"+(S.layer==="biome_country"?"":" off");
}}

// ── Populate filter selects ───────────────────────────────────
function populateFilters(){{
  // biomes
  const selB=document.getElementById("f-biome");
  selB.innerHTML="";
  Object.entries(BIOME_NAMES).sort((a,b)=>a[0]-b[0]).forEach(([code,name])=>{{
    const o=document.createElement("option");o.value=code;o.textContent=name;selB.appendChild(o);
  }});
  // realms
  const selR=document.getElementById("f-realm");
  selR.innerHTML="";
  Object.entries(REALM_NAMES).sort((a,b)=>a[1].localeCompare(b[1])).forEach(([code,name])=>{{
    const o=document.createElement("option");o.value=code;o.textContent=`${{name}} (${{code}})`;selR.appendChild(o);
  }});
  // countries — build name map, sort by full name, show full name only
  const countryMap={{}};
  LAYERS.biome_country.features.forEach(f=>{{
    const p=f.properties;
    if(p.ISO&&p.country_name&&p.country_name!=="No data"&&p.country_name!==p.ISO)
      countryMap[p.ISO]=p.country_name;
  }});
  const sortedCountries=Object.entries(countryMap)
    .sort((a,b)=>a[1].localeCompare(b[1]));
  const selC=document.getElementById("f-country");
  selC.innerHTML="";
  sortedCountries.forEach(([iso,name])=>{{
    const o=document.createElement("option");
    o.value=iso;
    o.textContent=name;
    selC.appendChild(o);
  }});
}}

// ── Animation ────────────────────────────────────────────────
let animTimer=null;
function stopAnim(){{
  if(animTimer){{clearInterval(animTimer);animTimer=null;}}
  const btn=document.getElementById("btn-play");
  if(btn) btn.textContent="\u25B6";
}}
function startAnim(){{
  stopAnim();
  animTimer=setInterval(()=>{{
    S.month=S.month===12?1:S.month+1;
    const sl=document.getElementById("month-sl");
    sl.value=S.month;
    document.getElementById("mlabel").textContent=MONTH_ABBR[S.month-1];
    refreshStyles();chartRanking();
  }},900);
  document.getElementById("btn-play").textContent="\u23F8";
}}

// ── Events ────────────────────────────────────────────────────
document.getElementById("sel-layer").addEventListener("change",e=>{{
  S.layer=e.target.value;S.selected=null;
  S.compareP=null;S.compareMode=false;
  document.getElementById("compare-label").textContent="";
  document.getElementById("btn-compare").textContent="+ Compare";
  document.getElementById("btn-clear-compare").style.display="none";
  syncFilters();buildMap(true);
  updateStatus(null);updateSummary(null);chartSeasonal(null);
}});
document.getElementById("sel-metric").addEventListener("change",e=>{{
  S.metric=e.target.value;
  const show=isMonthlyMetric();
  document.getElementById("month-grp").style.display=show?"flex":"none";
  if(!show) stopAnim();
  applyMetricAccent();
  refreshStyles();chartRanking();
}});
document.getElementById("btn-play").addEventListener("click",()=>{{
  if(animTimer) stopAnim(); else startAnim();
}});
document.getElementById("month-sl").addEventListener("input",e=>{{
  S.month=+e.target.value;
  document.getElementById("mlabel").textContent=MONTH_ABBR[S.month-1];
  refreshStyles();chartRanking();
}});
document.getElementById("btn-reset").addEventListener("click",()=>{{
  S.layer="biome";S.metric="annual_mean_km2";S.month=1;
  S.fBiomes=[];S.fRealms=[];S.fCountries=[];S.selected=null;
  document.getElementById("sel-layer").value="biome";
  document.getElementById("sel-metric").value="annual_mean_km2";
  document.getElementById("month-sl").value=1;
  document.getElementById("mlabel").textContent="Jan";
  document.getElementById("month-grp").style.display="none";
  stopAnim();
  applyMetricAccent();
  ["f-biome","f-realm","f-country"].forEach(id=>
    Array.from(document.getElementById(id).options).forEach(o=>{{o.selected=false;}}));
  setOverlay("");
  document.getElementById("sel-overlay").value="";
  document.getElementById("sel-layer").disabled=false;
  document.getElementById("sel-layer").title="";
  syncFilters();buildMap(true);
  updateStatus(null);updateSummary(null);chartSeasonal(null);chartRanking();chartComp();
}});
["f-biome","f-realm","f-country"].forEach(id=>
  document.getElementById(id).addEventListener("change",()=>{{
    S.fBiomes=selVals("f-biome");
    S.fRealms=S.layer==="biome_realm"?selVals("f-realm"):[];
    S.fCountries=S.layer==="biome_country"?selVals("f-country"):[];
    buildMap(true);chartRanking();chartComp();
  }}));
document.getElementById("btn-clear").addEventListener("click",()=>{{
  S.fBiomes=[];S.fRealms=[];S.fCountries=[];
  ["f-biome","f-realm","f-country"].forEach(id=>
    Array.from(document.getElementById(id).options).forEach(o=>{{o.selected=false;}}));
  buildMap(true);chartRanking();chartComp();
}});

// ── Country search ───────────────────────────────────────────
function doCountrySearch(){{
  const q=document.getElementById("country-search").value.trim().toLowerCase();
  const res=document.getElementById("search-results");
  if(!q){{res.textContent="";return;}}

  // collect all unique countries across layers with their bounds
  const matches=[];
  const seen=new Set();
  LAYERS.biome_country.features.forEach(f=>{{
    const p=f.properties;
    if(!p.ISO||seen.has(p.ISO)) return;
    const iso=(p.ISO||"").toLowerCase();
    const name=(p.country_name&&p.country_name!=="No data"?p.country_name:p.ISO).toLowerCase();
    if(iso.includes(q)||name.includes(q)){{
      seen.add(p.ISO);
      const displayName=p.country_name&&p.country_name!=="No data"?p.country_name:p.ISO;
      matches.push({{iso:p.ISO,name:displayName}});
    }}
  }});
  matches.sort((a,b)=>a.name.localeCompare(b.name));

  if(!matches.length){{res.textContent="No matches found.";return;}}

  if(matches.length===1){{
    zoomToCountry(matches[0].iso);
    res.textContent="";
    return;
  }}

  res.innerHTML=matches.slice(0,8).map(m=>
    `<span style="cursor:pointer;display:block;padding:2px 4px;border-radius:3px"
      onmouseover="this.style.background='#243548'" onmouseout="this.style.background=''"
      onclick="zoomToCountry('${{m.iso}}');document.getElementById('search-results').innerHTML='';
               document.getElementById('country-search').value='${{m.name}}'"
    >${{m.name}}</span>`
  ).join("")+(matches.length>8?`<span style="color:#4a6a82">+${{matches.length-8}} more</span>`:"");
}}

function zoomToCountry(iso){{
  const feats=LAYERS.biome_country.features.filter(f=>f.properties.ISO===iso);
  if(!feats.length){{document.getElementById("search-results").textContent="Not found in data.";return;}}

  // compute bounding box from all features for this country
  let minLat=90,maxLat=-90,minLon=180,maxLon=-180;
  feats.forEach(f=>{{
    const coords=[];
    const collect=c=>{{
      if(Array.isArray(c[0]))c.forEach(collect);
      else{{coords.push(c);}};
    }};
    collect(f.geometry.coordinates);
    coords.forEach(([lon,lat])=>{{
      if(lat<minLat)minLat=lat;if(lat>maxLat)maxLat=lat;
      if(lon<minLon)minLon=lon;if(lon>maxLon)maxLon=lon;
    }});
  }});

  map.fitBounds([[minLat,minLon],[maxLat,maxLon]],{{padding:[30,30]}});

  // highlight all features for this country briefly
  if(gjLayer){{
    gjLayer.eachLayer(l=>{{
      if(l.feature.properties.ISO===iso){{
        l.setStyle({{color:"#e8a45b",weight:2.5,fillOpacity:.85}});
        setTimeout(()=>refreshStyles(),2000);
      }}
    }});
  }}
}}

document.getElementById("btn-country-search").addEventListener("click",doCountrySearch);
document.getElementById("country-search").addEventListener("keydown",e=>{{
  if(e.key==="Enter") doCountrySearch();
}});
document.getElementById("country-search").addEventListener("input",doCountrySearch);

// ── Compare button ───────────────────────────────────────────
document.getElementById("btn-compare").addEventListener("click",()=>{{
  if(S.compareMode){{
    S.compareMode=false;
    document.getElementById("btn-compare").textContent="+ Compare";
    document.getElementById("btn-compare").classList.remove("active");
  }} else if(S.selected){{
    S.compareMode=true;
    document.getElementById("btn-compare").textContent="Cancel";
    document.getElementById("btn-compare").classList.add("active");
  }}
}});
document.getElementById("btn-clear-compare").addEventListener("click",()=>{{
  S.compareP=null;S.compareMode=false;
  document.getElementById("compare-label").textContent="";
  document.getElementById("btn-compare").textContent="+ Compare";
  document.getElementById("btn-compare").classList.remove("active");
  document.getElementById("btn-clear-compare").style.display="none";
  const primary=getSelectedProps();
  if(primary) chartSeasonal(primary);
  // remove compare note from insight panel
  const cn=document.getElementById("insight-panel")?.querySelector(".compare-note");
  if(cn) cn.remove();
  document.getElementById("compare-label").textContent="";
}});

// ── Pixel overlay system ─────────────────────────────────────
let overlayLayer   = null;
let overlayMeta    = null;
let olAnimTimer    = null;
let olMonth        = 1;
const MONTH_ABBR_S = ["Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec"];

async function loadOverlayMeta(){{
  try{{
    const r = await fetch("overlays/meta_overlays.json");
    overlayMeta = await r.json();
  }} catch(_) {{
    overlayMeta = null;  // overlays not generated yet
  }}
}}

function getOverlayBounds(){{
  if(!overlayMeta) return [[-90,-180],[90,180]];
  // use pre-computed Leaflet bounds from meta if available
  if(overlayMeta.leaflet_bounds) return overlayMeta.leaflet_bounds;
  const b = overlayMeta.bounds;
  return [[b.south, b.west],[b.north, b.east]];
}}

function setOverlay(id){{
  // remove existing
  if(overlayLayer){{ map.removeLayer(overlayLayer); overlayLayer=null; }}
  stopOlAnim();

  const legend   = document.getElementById("overlay-legend");
  const animBar  = document.getElementById("ol-anim-bar");

  if(!id){{
    // restore polygon layer — rebuild if needed
    legend.classList.remove("visible");
    animBar.classList.remove("visible");
    stopOlAnim();
    if(gjLayer && !map.hasLayer(gjLayer)){{
      gjLayer.addTo(map);
    }} else if(!gjLayer){{
      buildMap(false);
    }}
    return;
  }}

  // hide polygon layer while pixel overlay is active
  if(gjLayer && map.hasLayer(gjLayer)) map.removeLayer(gjLayer);

  if(!overlayMeta){{
    alert("Pixel overlays not found. Run prepare_pixel_overlays.py first.");
    document.getElementById("sel-overlay").value="";
    return;
  }}

  const layerMeta = overlayMeta.layers.find(l=>l.id===id);
  if(!layerMeta) return;

  const bounds = getOverlayBounds();

  if(id === "monthly"){{
    // animated monthly overlay
    animBar.classList.add("visible");
    showOlMonth(olMonth, bounds);
    buildOverlayLegend(layerMeta);
  }} else {{
    animBar.classList.remove("visible");
    overlayLayer = L.imageOverlay(layerMeta.file, bounds, {{opacity:0.75}});
    overlayLayer.addTo(map);
    buildOverlayLegend(layerMeta);
  }}
}}

function showOlMonth(m, bounds){{
  if(overlayLayer){{ map.removeLayer(overlayLayer); overlayLayer=null; }}
  if(!overlayMeta) return;
  const layerMeta = overlayMeta.layers.find(l=>l.id==="monthly");
  if(!layerMeta) return;
  const file = layerMeta.files[m-1];
  overlayLayer = L.imageOverlay(file, bounds||getOverlayBounds(), {{opacity:0.85}});
  overlayLayer.addTo(map);
  document.getElementById("ol-month-label").textContent = MONTH_ABBR_S[m-1];
  document.getElementById("ol-month-sl").value = m;

  // update area counter
  const stats = layerMeta.monthly_stats?.[m-1];
  const counter = document.getElementById("ol-area-counter");
  if(stats && counter){{
    const fmt = v => v>=1e6?(v/1e6).toFixed(2)+" M km²":v>=1e3?(v/1e3).toFixed(1)+" k km²":v.toFixed(0)+" km²";
    counter.innerHTML =
      `<span style="color:#253494">&#9632;</span> ${{fmt(stats.area_perennial_km2)}} &nbsp;`+
      `<span style="color:#41ab5d">&#9632;</span> ${{fmt(stats.area_seasonal_km2)}} &nbsp;`+
      `<span style="color:#fec44f">&#9632;</span> ${{fmt(stats.area_episodic_km2)}}`;
  }} else if(counter){{
    counter.textContent="";
  }}
}}

function buildOverlayLegend(layerMeta){{
  const el = document.getElementById("overlay-legend");
  const legend = layerMeta.legend || [];
  el.innerHTML = `<div class="ol-title">${{layerMeta.label}}</div>
    ${{legend.map(l=>`<div class="ol-row">
      <div class="ol-swatch" style="background:${{l.color}}"></div>
      <span>${{l.label}}</span></div>`).join("")}}`;
  el.classList.add("visible");
}}

function stopOlAnim(){{
  if(olAnimTimer){{ clearInterval(olAnimTimer); olAnimTimer=null; }}
  const btn = document.getElementById("btn-ol-play");
  if(btn) btn.textContent="▶";
}}

function startOlAnim(){{
  stopOlAnim();
  const bounds = getOverlayBounds();
  olAnimTimer = setInterval(()=>{{
    olMonth = olMonth===12?1:olMonth+1;
    showOlMonth(olMonth, bounds);
  }}, 800);
  document.getElementById("btn-ol-play").textContent="⏸";
}}

// Overlay controls
document.getElementById("sel-overlay").addEventListener("change", e=>{{
  const val = e.target.value;
  if(val){{
    // pixel overlays are independent of biome layer — switch to biome, disable realm/country filters
    if(S.layer !== "biome"){{
      S.layer = "biome";
      document.getElementById("sel-layer").value = "biome";
      syncFilters();
      buildMap(false);
    }}
    // disable the layer selector while a pixel overlay is active
    document.getElementById("sel-layer").disabled = true;
    document.getElementById("sel-layer").title = "Switch pixel layer to None to change map layer";
  }} else {{
    document.getElementById("sel-layer").disabled = false;
    document.getElementById("sel-layer").title = "";
  }}
  setOverlay(val);
}});
document.getElementById("btn-ol-play").addEventListener("click",()=>{{
  if(olAnimTimer) stopOlAnim(); else startOlAnim();
}});
document.getElementById("ol-month-sl").addEventListener("input",e=>{{
  olMonth=+e.target.value;
  stopOlAnim();
  showOlMonth(olMonth, getOverlayBounds());
}});

// ── Init ──────────────────────────────────────────────────────
(function(){{
  const n=GLOBAL_MEAN;
  const el=document.getElementById("gsv");
  const unit=document.querySelector(".gsu");
  if(n>=1e6){{ el.textContent=(n/1e6).toFixed(2); if(unit) unit.textContent="M km²"; }}
  else if(n>=1e3){{ el.textContent=(n/1e3).toFixed(1); if(unit) unit.textContent="k km²"; }}
  else {{ el.textContent=n.toFixed(0); if(unit) unit.textContent="km²"; }}
}})();
applyMetricAccent();
populateFilters();
loadOverlayMeta();
buildMap(true);
chartSeasonal(null);chartRanking();chartComp();
</script>
</body>
</html>"""

OUTPUT_HTML.write_text(HTML, encoding="utf-8")
size_mb = OUTPUT_HTML.stat().st_size / 1e6
print(f"Done. Written to {OUTPUT_HTML}  ({size_mb:.1f} MB)")
print("Share the file or drag it into any browser.")