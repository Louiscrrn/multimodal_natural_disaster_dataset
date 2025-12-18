import pandas as pd
from datetime import timedelta

from gdelt_loading_files import load_gdelt_index, get_files_for_day_3h
from gdelt_cyclone_pipeline import run_pipeline


def main(
    ibtracs_path="data/processed/ibtracs_era5_20251218_1715.csv",
    output_file="data/processed/cyclones_mentions_gdelt_3h_2022_2023.csv",
    start_year=2022,
    end_year=2023,
    window_days=3
):
    # -------------------------------
    # Load and filter cyclone data
    # -------------------------------
    df_cyclones = pd.read_csv(ibtracs_path)
    df_cyclones["Timestamp"] = pd.to_datetime(df_cyclones["Timestamp"])

    df_cyclones = df_cyclones[
        (df_cyclones["Timestamp"].dt.year >= start_year) &
        (df_cyclones["Timestamp"].dt.year <= end_year)
    ]

    # -------------------------------
    # Build ±window_days date range
    # -------------------------------
    all_dates_set = set()

    for ts in df_cyclones["Timestamp"]:
        window_range = [
            ts + timedelta(days=offset)
            for offset in range(-window_days, window_days + 1)
        ]
        all_dates_set.update(d.strftime("%Y%m%d") for d in window_range)

    active_dates = sorted(all_dates_set)

    print(f"Processing {len(active_dates)} unique dates "
          f"(±{window_days} days around cyclone timestamps)")

    # -------------------------------
    # Load GDELT index
    # -------------------------------
    gdelt_files = load_gdelt_index()

    # -------------------------------
    # Select GDELT files (3-hourly)
    # -------------------------------
    gdelt_mentions_files = []
    gdelt_gkg_files = []

    for date_str in active_dates:
        gdelt_mentions_files.extend(
            get_files_for_day_3h(date_str, gdelt_files, type="mentions.CSV")
        )
        gdelt_gkg_files.extend(
            get_files_for_day_3h(date_str, gdelt_files, type="gkg.csv")
        )

    print(f"GDELT mentions files: {len(gdelt_mentions_files)}")
    print(f"GDELT GKG files: {len(gdelt_gkg_files)}")

    # -------------------------------
    # Prepare cyclone dates
    # -------------------------------
    df_cyclones["date"] = df_cyclones["Timestamp"].dt.strftime("%Y%m%d")

    # -------------------------------
    # Run pipeline
    # -------------------------------
    df_all_articles = run_pipeline(
        df_cyclones,
        gdelt_mentions_files,
        gdelt_gkg_files,
        output_file=output_file
    )

    return df_all_articles


if __name__ == "__main__":
    main()
