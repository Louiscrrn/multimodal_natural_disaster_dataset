import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm
import cdsapi
from pathlib import Path
import time
import numpy as np
from datetime import datetime

# =====================
# CONFIG
# =====================

ERA5_OUT = Path("../data/raw/era5_yearly")
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

BASIN_BBOX = {
    "NA": [60, -100, 0, -10],
    "EP": [40, -160, 0, -80],
    "WP": [50, 100, 0, 180],
    "NI": [30, 40, -5, 100],
    "SI": [0, 20, -40, 100],
    "SP": [0, 160, -40, -120],
    "SA": [0, -60, -40, 20],
}

BASIN_MONTHS = {
    "NA": [6, 7, 8, 9, 10, 11],
    "EP": [6, 7, 8, 9, 10, 11],
    "NI": [5, 6, 8, 10, 11, 12],
    "SI": [12, 1, 2, 3, 4, 5],
    "SP": [12, 1, 2, 3],
    "WP": [4, 5, 6, 7, 8, 9, 10, 11, 12],
}

ERA5_HOURS = ["00:00", "03:00", "06:00", "09:00", "12:00", "15:00", "18:00", "21:00"]

# =====================
# UTILS
# =====================

def normalize_lon_0_360(lon):
    return np.mod(np.asarray(lon, dtype=float), 360.0)

def cds_day_list(days):
    return [f"{int(d):02d}" for d in sorted(set(days))]

def clean_basin(b):
    if isinstance(b, bytes):
        return b.decode("utf-8")
    if isinstance(b, str) and b.startswith("b'"):
        return b[2:-1]
    return b

# =====================
# LOAD IBTRACS
# =====================

def load_IBTrACS(path, years):
    df = pd.read_csv(path)
    df["time_stamp"] = pd.to_datetime(df["time_stamp"], errors="coerce")
    df = df.dropna(subset=["time_stamp"])
    df["basin"] = df["basin"].apply(clean_basin)
    df = df[
        (df["time_stamp"].dt.year >= years[0]) &
        (df["time_stamp"].dt.year <= years[1])
    ]
    print(
        "IBTrACS:",
        df["time_stamp"].dt.year.min(), "→", df["time_stamp"].dt.year.max(),
        "| obs:", len(df),
        "| cyclones:", df["sid"].nunique()
    )
    return df

# =====================
# ERA5 SAMPLING
# =====================

def sample_era5_year(nc_path, df):
    ds = xr.open_dataset(nc_path)
    time_dim = "time" if "time" in ds.dims else "valid_time"

    df = df.copy()
    df["lon_0360"] = normalize_lon_0_360(df["lon"].values)

    results = []

    for t, g in tqdm(df.groupby("time_stamp"), desc="Sampling ERA5"):
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
            .rename(columns={v_nc: v for v, v_nc in ERA5_VAR_MAP.items()})
        )

        results.append(pd.concat([g.reset_index(drop=True), era_df], axis=1))

    return pd.concat(results, ignore_index=True)

# =====================
# MAIN PIPELINE
# =====================

def run_year_basin(df, year, basin, force_download=False):

    df_y = df[(df["time_stamp"].dt.year == year) & (df["basin"] == basin)]
    if df_y.empty:
        raise ValueError("No data for this year/basin")

    out_nc = ERA5_OUT / f"era5_{year}_{basin}.nc"

    print(f"\nYEAR {year} | BASIN {basin}")
    print("obs:", len(df_y))
    print("cyclones:", df_y["sid"].nunique())
    print("bbox:", BASIN_BBOX[basin])

    if not out_nc.exists() or force_download:
        c = cdsapi.Client()
        req = {
            "product_type": "reanalysis",
            "variable": VARIABLES,
            "year": str(year),
            "month": [f"{m:02d}" for m in BASIN_MONTHS[basin]],
            "day": cds_day_list(df_y["time_stamp"].dt.day.unique()),
            "time": ERA5_HOURS,
            "area": BASIN_BBOX[basin],
            "grid": GRID,
            "data_format": "netcdf",
        }

        t0 = time.perf_counter()
        c.retrieve("reanalysis-era5-single-levels", req, str(out_nc))
        print(f"ERA5 download: {time.perf_counter() - t0:.1f}s")
    else:
        print("ERA5 file exists → skip download")

    return sample_era5_year(out_nc, df_y)

def run_year_all_basins(
    df,
    year,
    basins,
    force_download=False,
):
    all_results = []

    for basin in basins:
        try:
            df_basin = run_year_basin(
                df,
                year=year,
                basin=basin,
                force_download=force_download,
            )
            all_results.append(df_basin)

        except Exception as e:
            print(f"[WARNING] Basin {basin} failed: {e}")

    if not all_results:
        raise RuntimeError("No basin produced data")

    return pd.concat(all_results, ignore_index=True)


def haversine_km(lat1, lon1, lat2, lon2):
    """
    Compute great-circle distance (km) between two lat/lon points.
    """
    R = 6371.0  # Earth radius (km)

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


# =====================
# POST PROCESS
# =====================


def post_process_final(df):
    """
    Final post-processing for IBTrACS + ERA5 dataset.

    - Compute ERA5 spatial approximation error (km)
    - Convert ERA5 MSLP from Pa → hPa
    - Keep clean and consistent columns
    - Safe against pandas chained assignment warnings
    """

    df = df.copy()

    # -------------------------------------------------
    # 1. Spatial error ERA5 vs IBTrACS
    # -------------------------------------------------
    required_cols = {"lat", "lon", "latitude", "longitude"}
    if required_cols.issubset(df.columns):
        df["era5_spatial_error_km"] = haversine_km(
            df["lat"].values,
            df["lon"].values,
            df["latitude"].values,
            df["longitude"].values,
        )
    else:
        raise ValueError("Missing ERA5 latitude/longitude for spatial error")

    # -------------------------------------------------
    # 2. Pressure conversion (ERA5 Pa → hPa)
    # -------------------------------------------------
    if "mean_sea_level_pressure" in df.columns:
        df["mean_sea_level_pressure_hpa"] = (
            df["mean_sea_level_pressure"] / 100.0
        )

    # -------------------------------------------------
    # 3. Final clean column selection
    # -------------------------------------------------
    final_columns = [
        "sid",
        "name",
        "basin",
        "season",
        "time_stamp",
        "lat",
        "lon",
        "wind",
        "pressure",                     # IBTrACS (hPa)
        "storm_speed",
        "storm_dir",
        "2m_temperature",               # ERA5 (K)
        "mean_sea_level_pressure_hpa",  # ERA5 (hPa)
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "era5_spatial_error_km",
    ]

    final_columns = [c for c in final_columns if c in df.columns]
    df_final = df[final_columns].copy()

    # -------------------------------------------------
    # 4. Sort & reset index
    # -------------------------------------------------
    df_final = (
        df_final
        .sort_values("time_stamp")
        .reset_index(drop=True)
    )

    return df_final



# =====================
# RUN
# =====================

if __name__ == "__main__":

    PROCESSED_DIR = Path("../data/processed")
    RAW_DIR = Path("../data/raw")

    year = 2022
    basins=['EP', 'NA', 'NI', 'SI', 'SP', 'WP']

    now = datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M")
    output_file_name = f"ibtracs_era5_{timestamp_str}.csv"
    
    path_ibtracs = PROCESSED_DIR / "other/ibtracs_usa_20251216.csv"
    out_csv = PROCESSED_DIR / output_file_name

    df = load_IBTrACS(path_ibtracs, years=[2000, 2024])

    df_out = run_year_all_basins(
        df,
        year=year,
        basins=basins,
    )

    df_final = post_process_final(df_out)

    df_final.to_csv(out_csv, index=False)

    print("\n[OK] Dataset saved:", out_csv)
    print("Shape:", df_out.shape)
    print("Cyclones:", df_out["sid"].nunique())
