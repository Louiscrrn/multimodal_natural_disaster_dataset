import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import argparse

ERA5_DIR = Path("../data/raw/era5_yearly")
PROCESSED_DIR = Path("../data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

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

BASIN_BBOX = {
    "NA": [60, -100, 0, -10],
    "EP": [40, -160, 0, -80],
    "WP": [50, 100, 0, 180],
    "NI": [30, 40, -5, 100],
    "SI": [0, 20, -40, 100],
    "SP": [0, 160, -40, -120],  
}

RENAME_COLUMNS = {
    "sid": "Storm_ID",
    "name": "Storm_Name",
    "number" : "number",
    "usa_atcf_id" : "atcf_id",
    "basin": "Ocean_Basin",
    "season": "Year",
    "time_stamp": "Timestamp",
    "lat": "Latitude",
    "lon": "Longitude",
    "wind": "Observed_Wind_Max_Knots",
    "pressure": "Observed_Pressure_Min_mb",
    "storm_speed": "Storm_Speed_Knots",
    "storm_dir": "Storm_Direction_Deg",
    "2m_temperature": "ERA5_Temp_2m_Kelvin",
    "mean_sea_level_pressure_hpa": "ERA5_Pressure_MSL_hPa",
    "10m_u_component_of_wind": "ERA5_Wind_U_Component",
    "10m_v_component_of_wind": "ERA5_Wind_V_Component",
    "era5_spatial_error_km": "ERA5_Position_Error_km",
}



def normalize_lon_0_360(lon):
    return np.mod(np.asarray(lon, dtype=float), 360.0)

def wrap_lon180(lon):
    lon = np.asarray(lon, dtype=float)
    return ((lon + 180) % 360) - 180

def clean_basin(b):
    if isinstance(b, bytes):
        return b.decode("utf-8")
    if isinstance(b, str) and b.startswith("b'"):
        return b[2:-1]
    return b

def haversine_km_periodic_lon(lat1, lon1, lat2, lon2):
    """
    Haversine
    """
    R = 6371.0

    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))

    dlat = lat2 - lat1

    dlon = (lon2 - lon1 + np.pi) % (2 * np.pi) - np.pi

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def load_ibtracs(path, years):
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


def filter_ibtracs_in_bbox(df, bbox):
    """
    Filtre spatial sur bbox (lon en [-180,180]), gère le cas anti-méridien.
    """
    north, west, south, east = bbox

    lat = df["lat"].values
    lon = wrap_lon180(df["lon"].values)

    in_lat = (lat >= south) & (lat <= north)

    if west <= east:
        in_lon = (lon >= west) & (lon <= east)
    else:
        in_lon = (lon >= west) | (lon <= east)

    return df[in_lat & in_lon].copy()


def sample_era5_existing(nc_path, df):
    ds = xr.open_dataset(nc_path)
    print(
        nc_path.name,
        "ERA5 lon range:",
        float(ds.longitude.min()),
        float(ds.longitude.max())
        )

    time_dim = "time" if "time" in ds.dims else "valid_time"

    lon_min = float(ds.longitude.min())
    lon_max = float(ds.longitude.max())

    era5_is_0360 = lon_max > 180

    df = df.copy()

    if era5_is_0360:
        # ERA5 in [0,360]
        df["lon_for_sampling"] = normalize_lon_0_360(df["lon"].values)
    else:
        # ERA5 in [-180,180]
        df["lon_for_sampling"] = wrap_lon180(df["lon"].values)

    results = []

    for t, g in tqdm(df.groupby("time_stamp"), desc=f"Sampling {nc_path.name}"):
        ds_t = ds.sel({time_dim: t}, method="nearest")

        sampled = ds_t.sel(
            latitude=xr.DataArray(g["lat"].values, dims="point"),
            longitude=xr.DataArray(g["lon_for_sampling"].values, dims="point"),
            method="nearest",
        )

        era_df = (
            sampled[[ERA5_VAR_MAP[v] for v in VARIABLES]]
            .to_dataframe()
            .reset_index(drop=True)
            .rename(columns={v_nc: v for v, v_nc in ERA5_VAR_MAP.items()})
        )

        #results.append(pd.concat([g.reset_index(drop=True), era_df], axis=1))
        g_reset = g.reset_index(drop=True)

        # éviter toute collision de colonnes (ex: number, season, etc.)
        era_df = era_df.loc[:, ~era_df.columns.isin(g_reset.columns)]

        results.append(pd.concat([g_reset, era_df], axis=1))


    return pd.concat(results, ignore_index=True)



def post_process_final(df):
    df = df.copy()

    required = {"lat", "lon", "latitude", "longitude"}
    if not required.issubset(df.columns):
        raise ValueError(f"Colonnes manquantes pour distance: {required - set(df.columns)}")

    lon_ib = wrap_lon180(df["lon"].values)
    lon_era = wrap_lon180(df["longitude"].values)

    df["era5_spatial_error_km"] = haversine_km_periodic_lon(
        df["lat"].values,
        lon_ib,
        df["latitude"].values,
        lon_era,
    )

    if "mean_sea_level_pressure" in df.columns:
        df["mean_sea_level_pressure_hpa"] = df["mean_sea_level_pressure"] / 100.0

    if "number.1" in df.columns:
        df = df.drop(columns=["number.1"])
        
    final_columns = [
        "sid", "name", "basin", "season", "time_stamp", "number", "usa_atcf_id",
        "lat", "lon", "wind", "pressure",
        "storm_speed", "storm_dir",
        "2m_temperature", "mean_sea_level_pressure_hpa",
        "10m_u_component_of_wind", "10m_v_component_of_wind",
        "era5_spatial_error_km",
        "latitude", "longitude",
    ]
    final_columns = [c for c in final_columns if c in df.columns]

    df_final = (
        df[final_columns]
        .sort_values("time_stamp")
        .reset_index(drop=True)
    )

    worst = df_final["era5_spatial_error_km"].max()
    print(f"[CHECK] max era5_spatial_error_km = {worst:.2f} km")
    if worst > 2000:
        print("[WARNING] Encore des erreurs très hautes. Voici 10 pires lignes :")
        cols = [c for c in ["time_stamp","basin","lat","lon","latitude","longitude","era5_spatial_error_km"] if c in df_final.columns]
        print(df_final.sort_values("era5_spatial_error_km", ascending=False)[cols].head(10).to_string(index=False))

    df_final = df_final.rename(columns=RENAME_COLUMNS)
    return df_final

def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample ERA5 data on IBTrACS cyclone tracks and post-process dataset"
    )

    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2022, 2023, 2024],
        help="Years to process (e.g. --years 2022 2023)"
    )

    parser.add_argument(
        "--basins",
        type=str,
        nargs="+",
        default=["EP", "NA", "NI", "SI", "SP", "WP"],
        help="Basins to process (e.g. --basins NA EP WP)"
    )

    parser.add_argument(
        "--input-ibtracs",
        type=str,
        default=str(PROCESSED_DIR / "other/ibtracs_processed_20251218_004636.csv"),
        help="Path to IBTrACS CSV file"
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    years = args.years
    basins = args.basins
    path_ibtracs = Path(args.input_ibtracs)
    print("Years :", years)
    print("Basins:", basins)
    print("IBTrACS:", path_ibtracs)

    df_ib = load_ibtracs(path_ibtracs, years=[min(years), max(years)])

    all_results = []

    for year in years:
        for basin in basins:

            nc_path = ERA5_DIR / f"era5_{year}_{basin}.nc"
            if not nc_path.exists():
                print(f"[SKIP] Missing {nc_path.name}")
                continue

            
            df_yb = df_ib[(df_ib["time_stamp"].dt.year == year) & (df_ib["basin"] == basin)].copy()
            df_yb = filter_ibtracs_in_bbox(df_yb, BASIN_BBOX[basin])

            if df_yb.empty:
                print(f"[SKIP] No IBTrACS after bbox filter {year} {basin}")
                continue

            print(basin, year, "IBTrACS lon min/max:", float(df_yb["lon"].min()), float(df_yb["lon"].max()), "| n:", len(df_yb))

            df_sampled = sample_era5_existing(nc_path, df_yb)
            all_results.append(df_sampled)

    df_all = pd.concat(all_results, ignore_index=True)
    df_final = post_process_final(df_all)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_csv = PROCESSED_DIR / f"ibtracs_era5_{timestamp}.csv"
    df_final.to_csv(out_csv, index=False)

    print("\n[OK] Final dataset saved:", out_csv)
    print("Shape:", df_final.shape)
    print("Cyclones:", df_final["Storm_ID"].nunique())