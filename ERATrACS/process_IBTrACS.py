import xarray as xr
import pandas as pd
from pathlib import Path
from datetime import datetime
import argparse


DEFAULT_VARS = [
    "sid", "name", "season", "number", "basin", "usa_atcf_id",
    "time",
    "usa_lat", "usa_lon",
    "usa_wind", "usa_pres",
    "storm_speed", "storm_dir"
]


def process_ibtracs_data(input_file, output_dir, variables):
    print(f"Loading data from: {input_file}")
    ds = xr.open_dataset(input_file)

    # Vérification des variables demandées
    missing = set(variables) - set(ds.variables)
    if missing:
        raise ValueError(f"Variables absentes dans le NetCDF : {missing}")

    ds_tc = ds[variables]

    print("Filtering records with missing USA lat/lon...")
    ds_tc = ds_tc.where(
        ds_tc["usa_lat"].notnull() & ds_tc["usa_lon"].notnull(),
        drop=True
    )

    print("Converting to pandas DataFrame...")
    df = ds_tc.to_dataframe().reset_index()

    print("Cleaning columns...")
    drop_cols = [c for c in ["storm", "lat", "lon"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    rename_map = {
        "usa_lat": "lat",
        "usa_lon": "lon",
        "usa_wind": "wind",
        "usa_pres": "pressure",
        "time": "time_stamp",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    print("Processing timestamps...")
    df["time_stamp"] = pd.to_datetime(df["time_stamp"])
    df["year"] = df["time_stamp"].dt.year
    df["month"] = df["time_stamp"].dt.month
    df["day"] = df["time_stamp"].dt.day
    df["hour"] = df["time_stamp"].dt.hour

    df = df.sort_values(by=["time_stamp", "sid"], ascending=[False, True])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = output_dir / f"ibtracs_processed_{ts}.csv"

    print(f"Saving to {out_file}")
    df.to_csv(out_file, index=False)

    print("Done.")
    print("Rows:", len(df))
    print("Cyclones:", df["sid"].nunique())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process IBTrACS NetCDF and export selected variables to CSV"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=False,
        default="../data/raw/IBTrACS.ALL.v04r01.nc",
        help="Path to IBTrACS NetCDF file"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=False,
        default="../data/raw/",
        help="Directory to save processed CSV"
    )

    parser.add_argument(
        "--vars",
        type=str,
        nargs="+",
        default=DEFAULT_VARS,
        help=(
            "Variables to extract from IBTrACS NetCDF. "
            "Default: USA-based best-track variables."
        )
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    process_ibtracs_data(
        input_file=args.input,
        output_dir=args.output_dir,
        variables=args.vars,
    )
