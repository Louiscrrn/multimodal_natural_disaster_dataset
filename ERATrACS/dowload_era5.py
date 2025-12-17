import pandas as pd
import cdsapi
from pathlib import Path
import time
import argparse


ERA5_OUT = Path("../data/raw/era5_yearly")
ERA5_OUT.mkdir(parents=True, exist_ok=True)

VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "mean_sea_level_pressure",
]

GRID = (0.5, 0.5)

ERA5_HOURS = [
    "00:00", "03:00", "06:00", "09:00",
    "12:00", "15:00", "18:00", "21:00"
]

BASIN_BBOX = {
    "NA": [60, -100, 0, -10],
    "EP": [40, -160, 0, -80],
    "WP": [50, 100, 0, 180],
    "NI": [30, 40, -5, 100],
    "SI": [0, 20, -40, 100],
    "SP": [0, 160, -40, -120],
}

BASIN_MONTHS = {
    "NA": [6, 7, 8, 9, 10, 11],
    "EP": [6, 7, 8, 9, 10, 11],
    "NI": [5, 6, 8, 10, 11, 12],
    "SI": [12, 1, 2, 3, 4, 5],
    "SP": [12, 1, 2, 3],
    "WP": [4, 5, 6, 7, 8, 9, 10, 11, 12],
}

def cds_day_list(days):
    return [f"{int(d):02d}" for d in sorted(set(days))]

def clean_basin(b):
    if isinstance(b, bytes):
        return b.decode("utf-8")
    if isinstance(b, str) and b.startswith("b'"):
        return b[2:-1]
    return b

def load_ibtracs_for_calendar(path, year, basin):
    df = pd.read_csv(path)
    df["time_stamp"] = pd.to_datetime(df["time_stamp"], errors="coerce")
    df = df.dropna(subset=["time_stamp"])
    df["basin"] = df["basin"].apply(clean_basin)

    df_y = df[
        (df["time_stamp"].dt.year == year) &
        (df["basin"] == basin)
    ]

    if df_y.empty:
        return None

    return df_y

def download_era5_year_basin(
    year,
    basin,
    df_calendar,
    force=False,
):
    out_nc = ERA5_OUT / f"era5_{year}_{basin}.nc"

    if out_nc.exists() and not force:
        print(f"[SKIP] {out_nc.name} already exists")
        return

    print(f"\nDownloading ERA5 | Year {year} | Basin {basin}")
    print("BBox:", BASIN_BBOX[basin])
    print("Months:", BASIN_MONTHS[basin])

    days = cds_day_list(df_calendar["time_stamp"].dt.day.unique())

    c = cdsapi.Client()

    request = {
        "product_type": "reanalysis",
        "variable": VARIABLES,
        "year": str(year),
        "month": [f"{m:02d}" for m in BASIN_MONTHS[basin]],
        "day": days,
        "time": ERA5_HOURS,
        "area": BASIN_BBOX[basin],
        "grid": GRID,
        "data_format": "netcdf",
    }

    t0 = time.perf_counter()
    c.retrieve(
        "reanalysis-era5-single-levels",
        request,
        str(out_nc),
    )
    print(f"[OK] Saved {out_nc.name} ({time.perf_counter() - t0:.1f}s)")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download ERA5 data by year and basin using IBTrACS calendar"
    )

    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2022, 2023, 2024],
        help="Years to download (e.g. --years 2022 2023)"
    )

    parser.add_argument(
        "--basins",
        type=str,
        nargs="+",
        default=["EP", "NA", "NI", "SI", "SP", "WP"],
        help="Basins to download (e.g. --basins NA EP WP)"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files already exist"
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    PROCESSED_DIR = Path("../data/processed")
    path_ibtracs = PROCESSED_DIR / "other/ibtracs_usa_20251216.csv"

    years = args.years
    basins = args.basins
    force = args.force

    print("Years :", years)
    print("Basins:", basins)
    print("Force :", force)

    for year in years:
        for basin in basins:
            if basin not in BASIN_BBOX:
                print(f"[SKIP] Unknown basin {basin}")
                continue

            df_cal = load_ibtracs_for_calendar(
                path_ibtracs,
                year=year,
                basin=basin,
            )

            if df_cal is None:
                print(f"[SKIP] No IBTrACS data for {year} {basin}")
                continue

            n_obs = len(df_cal)
            n_cyclones = df_cal["sid"].nunique()
            period_start = df_cal["time_stamp"].min()
            period_end = df_cal["time_stamp"].max()

            print(
                f"[IBTrACS] {year} {basin} | "
                f"cyclones: {n_cyclones} | "
                f"observations: {n_obs} | "
                f"period: {period_start} → {period_end}"
                )

            download_era5_year_basin(
                year=year,
                basin=basin,
                df_calendar=df_cal,
                force=force,
            )

    print("\n[DONE] ERA5 download pipeline finished")