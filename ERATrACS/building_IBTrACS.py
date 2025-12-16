import xarray as xr
import pandas as pd
from pathlib import Path
from datetime import datetime
import os 

def process_ibtracs_data(input_file: str, output_dir: str):
    """
    Loads IBTrACS data from a NetCDF file, filters it, cleans it, and saves 
    the resulting DataFrame to a CSV file.

    Args:
        input_file (str): Path to the IBTrACS NetCDF file.
        output_dir (str): Directory where the processed CSV file should be saved.
    """
    try:
        # 1. Load the Dataset
        print(f"Loading data from: {input_file}")
        ds = xr.open_dataset(input_file)
        
        # 2. Select relevant variables (focusing on US estimates)
        vars_usa = [
            "sid", "name", "season", "basin",
            "time",
            "usa_lat", "usa_lon",
            "usa_wind", "usa_pres",
            "storm_speed", "storm_dir"
        ]
        ds_tc = ds[vars_usa]
        
        # 3. Filter to remove records where position (lat/lon) is NaN 
        # (based on US estimation source)
        print("Filtering out records with missing position (NaN)...")
        ds_tc_filtered = ds_tc.where(
            ds_tc["usa_lat"].notnull()
            & ds_tc["usa_lon"].notnull(),
            drop=True
        )
        
        # 4. Convert to a Pandas DataFrame
        print("Converting to Pandas DataFrame...")
        df_tc_pd = (
            ds_tc_filtered
            .to_dataframe()
            .reset_index()
        )
        
        # 5. Clean up and Rename columns
        print("Cleaning up and renaming columns...")
        df_tc_pd = (
            df_tc_pd
            # Drop global lat/lon (redundant with usa_lat/usa_lon) and the storm index
            .drop(columns=["storm", "lat", "lon"])
            .rename(columns={
                "usa_lat": "lat",
                "usa_lon": "lon",
                "usa_wind": "wind",
                "usa_pres": "pressure",
                "time": "time_stamp",
            })
            # Final check to drop any rows still missing lat/lon
            .dropna(subset=["lat", "lon"])
            .reset_index(drop=True)
        )
        
        # 6. Convert timestamp and extract time components
        print("Processing and extracting time components...")
        df_tc_pd["time_stamp"] = pd.to_datetime(df_tc_pd["time_stamp"])
        df_tc_pd["year"] = df_tc_pd["time_stamp"].dt.year
        df_tc_pd["month"] = df_tc_pd["time_stamp"].dt.month
        df_tc_pd["day"] = df_tc_pd["time_stamp"].dt.day
        df_tc_pd["hour"] = df_tc_pd["time_stamp"].dt.hour
        
        # 7. Sort by most recent timestamp
        df = df_tc_pd.sort_values(
            by=["time_stamp", "sid"],
            ascending=[False, True]
        )

        # 8. Save the DataFrame
        
        # Create the full output path (e.g., ./data/processed/)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True) # Create directory if it doesn't exist

        # Generate the timestamp for the filename
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"ibtracs_usa_processed_{current_datetime}.csv"
        full_output_file = output_path / file_name

        print(f"Saving DataFrame to: {full_output_file}")
        df.to_csv(full_output_file, index=False)
        print("Processing completed successfully.")
        print(f"Final number of rows: {len(df)}")

    except FileNotFoundError:
        print(f"ERROR: Input file not found at location: {input_file}")
    except Exception as e:
        print(f"An error occurred during processing: {e}")


if __name__ == "__main__":
    # Define paths
    # The source NetCDF file is expected to be in './data/'
    INPUT_FILE = "../data/raw/IBTrACS.ALL.v04r01.nc"
    
    # The output directory is specified as './data/processed/'
    # The relative path is based on where the script is executed.
    OUTPUT_DIR = "../data/processed/" 
    
    process_ibtracs_data(INPUT_FILE, OUTPUT_DIR)