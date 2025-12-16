import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm

import cdsapi
from pathlib import Path
import time

ERA5_OUT = Path("../data/raw/era5_yearly_tests")
ERA5_OUT.mkdir(parents=True, exist_ok=True)

VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "mean_sea_level_pressure",
]

ERA5_VAR_MAP = {
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "2m_temperature": "t2m",
    "mean_sea_level_pressure": "msl",
}

GRID = (0.5, 0.5)      
BBOX_PAD = 2.0        

def normalize_lon_0_360(lon):
    lon = np.asarray(lon, dtype=float)
    return np.mod(lon, 360.0)

def compute_bbox(df, pad=BBOX_PAD):
    lat_min = float(df["lat"].min()) - pad
    lat_max = float(df["lat"].max()) + pad

    lon = normalize_lon_0_360(df["lon"].values)
    lon_min = float(lon.min()) - pad
    lon_max = float(lon.max()) + pad

    return [
        float(min(90.0, lat_max)),
        float(max(0.0, lon_min)),
        float(max(-90.0, lat_min)),
        float(min(360.0, lon_max)),
    ]  # [N, W, S, E]

def cds_day_list(days):
    return [f"{int(d):02d}" for d in sorted(set(days))]

def cds_time_list(hours):
    return [f"{int(h):02d}:00" for h in sorted(set(hours))]

def subset_top_cyclones_per_year(df, year, n=50):
    df_y = df[df["time_stamp"].dt.year == year].copy()

    sids = (
        df_y.groupby("sid")
            .size()
            .sort_values(ascending=False)
            .head(n)
            .index
    )

    return df_y[df_y["sid"].isin(sids)].copy()

def sample_era5_to_obs(nc_path, df):
    ds = xr.open_dataset(nc_path)
    time_dim = "time" if "time" in ds.dims else "valid_time"

    df = df.copy()
    df["lon_0360"] = normalize_lon_0_360(df["lon"].values)

    p_time = xr.DataArray(df["time_stamp"].values, dims="point")
    p_lat = xr.DataArray(df["lat"].values, dims="point")
    p_lon = xr.DataArray(df["lon_0360"].values, dims="point")

    ds_t = ds.sel({time_dim: p_time}, method="nearest")
    ds_p = ds_t.sel(latitude=p_lat, longitude=p_lon, method="nearest")

    era_vars_nc = [ERA5_VAR_MAP[v] for v in VARIABLES]
    era_df = ds_p[era_vars_nc].to_dataframe().reset_index(drop=True)
    era_df = era_df.rename(columns={v_nc: v for v, v_nc in ERA5_VAR_MAP.items()})

    return pd.concat([df.reset_index(drop=True), era_df], axis=1)


def sample_era5_year_fast(nc_path, df_ibtracs):

    ds = xr.open_dataset(nc_path)
    time_dim = "time" if "time" in ds.dims else "valid_time"

    df = df_ibtracs.copy()
    df["time_stamp"] = pd.to_datetime(df["time_stamp"])
    df["lon_0360"] = normalize_lon_0_360(df["lon"].values)

    results = []

    for t, g in tqdm(df.groupby("time_stamp"), desc="Sampling ERA5"):

        # ✅ nearest temporal match (CRITICAL FIX)
        ds_t = ds.sel({time_dim: t}, method="nearest")

        sampled = ds_t.sel(
            latitude=xr.DataArray(g["lat"].values, dims="point"),
            longitude=xr.DataArray(g["lon_0360"].values, dims="point"),
            method="nearest",
        )

        era_df = (
            sampled[[ERA5_VAR_MAP[v] for v in VARIABLES]]
            .to_dataframe()
            .reset_index(drop=True)
        )

        era_df = era_df.rename(
            columns={v_nc: v for v, v_nc in ERA5_VAR_MAP.items()}
        )

        merged = pd.concat(
            [g.reset_index(drop=True), era_df],
            axis=1
        )

        results.append(merged)

    # Safety
    if len(results) == 0:
        raise RuntimeError("No ERA5 samples extracted — check time alignment")

    return pd.concat(results, ignore_index=True)

def split_by_basin(df_year):
    return dict(tuple(df_year.groupby("basin")))

def subset_basins(df, basins):
    return df[df["basin"].isin(basins)].copy()

def run_yearly_test_by_basins(
    df,
    year,
    basins,
    n_cyclones=50,
    force_download=False,
):
    """
    ERA5 yearly pipeline with basin control.

    - 1 ERA5 request per basin
    - Top-N cyclones per basin define the bbox
    - ERA5 sampled on ALL IBTrACS obs of the basin/year
    """

    df = df.copy()
    df["time_stamp"] = pd.to_datetime(df["time_stamp"])

    # --- Subset year
    df_year = df[df["time_stamp"].dt.year == year].copy()
    if df_year.empty:
        raise ValueError(f"No IBTrACS data for year {year}")

    # --- Subset basins
    df_year = subset_basins(df_year, basins)

    if df_year.empty:
        raise ValueError(f"No data for basins {basins} in {year}")

    merged_all = []

    print(f"\n=== YEAR {year} | BASINS {basins} ===")

    for basin in basins:
        print(f"\n--- Basin {basin} ---")

        df_basin = df_year[df_year["basin"] == basin].copy()
        if df_basin.empty:
            print("No data → skipped")
            continue

        # Top-N cyclones for bbox
        df_bbox = (
            df_basin.groupby("sid")
            .size()
            .sort_values(ascending=False)
            .head(n_cyclones)
            .index
        )
        df_bbox = df_basin[df_basin["sid"].isin(df_bbox)]

        area = compute_bbox(df_bbox)

        days   = df_basin["time_stamp"].dt.day.unique()
        hours  = df_basin["time_stamp"].dt.hour.unique()
        months = df_basin["time_stamp"].dt.month.unique()

        out_nc = ERA5_OUT / f"era5_year{year}_{basin}_n{n_cyclones}.nc"

        print("obs:", len(df_basin))
        print("cyclones:", df_basin["sid"].nunique())
        print("bbox:", area)

        # --- ERA5 request (1 per basin)
        if not out_nc.exists() or force_download:
            c = cdsapi.Client()
            req = {
                "product_type": "reanalysis",
                "variable": VARIABLES,
                "year": str(year),
                "month": [f"{m:02d}" for m in sorted(months)],
                "day": cds_day_list(days),
                "time": cds_time_list(hours),
                "area": area,
                "grid": [GRID[0], GRID[1]],
                "data_format": "netcdf",
            }

            t0 = time.perf_counter()
            c.retrieve("reanalysis-era5-single-levels", req, str(out_nc))
            print(f"ERA5 download: {time.perf_counter() - t0:.1f}s")
        else:
            print("ERA5 file exists → skip download")

        # --- FAST sampling
        print("Sampling ERA5...")
        merged_basin = sample_era5_year_fast(out_nc, df_basin)
        merged_all.append(merged_basin)

    if not merged_all:
        raise RuntimeError("No basin produced data")

    return pd.concat(merged_all, ignore_index=True)

def clean_basin(b):
    if isinstance(b, str):
        if b.startswith("b'") and b.endswith("'"):
            return b[2:-1]
        else:
            return b
    if isinstance(b, bytes):
        return b.decode("utf-8")
    return b


def load_IBTrACS(path, years=[2000, 2024], verbose=True):

    df = pd.read_csv(path)

    df["time_stamp"] = pd.to_datetime(df["time_stamp"], errors="coerce")
    df = df.dropna(subset=["time_stamp"])

    df["basin"] = df["basin"].apply(clean_basin)

    df = df[
        (df["time_stamp"].dt.year >= years[0]) &
        (df["time_stamp"].dt.year <= years[1])
    ].copy()

    if verbose:
        print(
            "Période retenue:",
            df["time_stamp"].dt.year.min(),
            "→",
            df["time_stamp"].dt.year.max(),
            "| obs:", len(df),
            "| cyclones:", df["sid"].nunique(),
            "| basins:", sorted(df["basin"].unique())
        )

    return df


if __name__ == "__main__":

    year = 2024
    n_cyclones = 3

    PROCESSED_DIR = Path("../data/processed")
    RAW_DIR = Path("../data/raw")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    path_ibtracs = PROCESSED_DIR / "ibtracs_usa_processed_20251216.csv"
    out_path = RAW_DIR / f"ibtracs_era5_{year}.csv"

    df = load_IBTrACS(path_ibtracs, years=[2000, 2024])

    df_2024 = run_yearly_test_by_basins(
        df,
        year=2024,
        basins=["NA"],
        n_cyclones=3,
    )

    df_2024.to_csv(out_path, index=False)

    print(f"[OK] Saved ERA5+IBTrACS dataset to: {out_path}")
    print("Shape:", df_2024.shape)
    print("Cyclones:", df_2024["sid"].nunique())
    print(
        "Period:",
        df_2024["time_stamp"].min(),
        "→",
        df_2024["time_stamp"].max(),
    )
