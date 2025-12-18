"""
TC-PRIMED Dataset Builder - Hierarchical Zarr Dataset with News Data
Creates a unified Zarr dataset combining IBTrACS/ERA5 tabular data, satellite imagery, and daily news

Features:
- Masks sentinel values (-9999, 0.0 for IR) as NaN BEFORE resizing
- Prevents interpolation artifacts from invalid data
- Resizes all images to 224×224 using bilinear interpolation
- Stores raw physical values (not normalized)
- Broadcasts daily news data to all 3-hourly timestamps

Architecture:
TC_Clean_Dataset.zarr/
└── {ATCF_ID}/
    ├── timestamps          # 1D: [datetime64]
    ├── tabular/           # All CSV features as 1D arrays
    ├── images_2d/         # 3D: (Time, 224, 224) - all satellite products
    ├── quality_meta/      # Match quality metrics
    └── news/              # Daily news data broadcasted to timestamps
        └── num_articles   # 1D: article count per timestamp

USAGE: python tc3.py
"""

import pandas as pd
import numpy as np
import s3fs
import xarray as xr
import zarr
import os
import re
from datetime import timedelta
from scipy.ndimage import zoom

# Configuration
INPUT_CSV = '../data/processed/small_15_ibtracs_era5_20251218_1520_reliefweb.csv'
NEWS_CSV = '../data/processed/cyclones_mentions_gdelt_3h_2022_2023.csv'  # NEW: News data path
OUTPUT_ZARR = 'TC_Clean_Dataset.zarr'
S3_PATH = 'noaa-nesdis-tcprimed-pds/v01r01/final'
TARGET_SIZE = (224, 224)
TIME_TOLERANCE_HOURS = 1.5

# GPROF subgroups and variable mappings
GPROF_SUBGROUPS = ['S1', 'S2', 'S3']
VAR_MAP = {
    'ir': ['IRWIN', 'brightness_temperature'],
    'surface_precip': ['surfacePrecipitation'],
    'convective_precip': ['convectivePrecipitation'],
    'rain_water_path': ['rainWaterPath'],
    'cloud_water_path': ['cloudWaterPath'],
    'ice_water_path': ['iceWaterPath']
}

# Pre-defined expected variables (prevents First Match Bug)
EXPECTED_2D_VARS = list(VAR_MAP.keys())


def safe_scalar(value):
    """Safely convert numpy array to Python scalar"""
    if isinstance(value, np.ndarray):
        return float(value.flat[0] if value.size > 0 else np.nan)
    return float(value)


def mask_sentinel_values(data, var_type='gprof'):
    """
    Mask sentinel/fill values as NaN before processing.
    This prevents interpolation from spreading invalid values.
    """
    data = data.astype(np.float32)
    
    # Common sentinel values in TC-PRIMED
    if var_type == 'ir':
        # IR: 0.0 is space/missing, values should be ~180-320K
        data[data == 0.0] = np.nan
        data[data < 100.0] = np.nan  # Any temp below 100K is invalid
    else:  # GPROF products
        # Mask standard sentinel values
        data[data == -9999.0] = np.nan
        data[data < -1000.0] = np.nan  # Catch other large negative sentinels
    
    return data


def resize_image(img, target_shape):
    """Resize 2D image to target shape using bilinear interpolation"""
    if img.shape == target_shape:
        return img
    zoom_factors = (target_shape[0] / img.shape[0], target_shape[1] / img.shape[1])
    return zoom(img, zoom_factors, order=1)


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in km between two points"""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def parse_timestamp_from_filename(filename):
    """Extract timestamp from TC-PRIMED filename"""
    match = re.search(r'_(\d{14})\.nc$', filename)
    return pd.to_datetime(match.group(1), format='%Y%m%d%H%M%S') if match else None


def get_variable(ds, var_names):
    """Get first available variable from list"""
    for var in var_names:
        if var in ds.variables:
            return var
    return None


def extract_2d_slice(data):
    """Extract 2D slice from xarray data by removing singleton/time dimensions"""
    while data.ndim > 2:
        # Remove first dimension if it's singleton or time
        if data.shape[0] == 1 or (hasattr(data, 'dims') and data.dims[0] in ['time', 'layer']):
            data = data[0] if isinstance(data, np.ndarray) else data.isel({data.dims[0]: 0})
        else:
            break
    return data.values if hasattr(data, 'values') else data


def align_news_data(storm_df, news_df, storm_id):
    """
    Align daily news data with 3-hourly storm timestamps.
    
    Broadcasts each daily news value to all timestamps within that day.
    This ensures every training sample has a complete feature vector.
    
    Args:
        storm_df: DataFrame with storm timestamps (3-hourly)
        news_df: DataFrame with daily news data
        storm_id: Storm ID to filter news data
        
    Returns:
        Array of article counts aligned with storm timestamps
    """
    # Filter news for this specific storm
    storm_news = news_df[news_df['Storm_ID'] == storm_id].copy()
    
    if len(storm_news) == 0:
        # No news for this storm - return zeros
        return np.zeros(len(storm_df), dtype=np.int32)
    
    # Convert day column to datetime (format: YYYYMMDD)
    storm_news['date'] = pd.to_datetime(storm_news['day'], format='%Y%m%d')
    storm_news = storm_news.sort_values('date').reset_index(drop=True)
    
    # Create aligned array
    storm_timestamps = pd.to_datetime(storm_df['Timestamp'])
    aligned_articles = np.zeros(len(storm_timestamps), dtype=np.int32)
    
    # Forward fill: for each timestamp, find the corresponding day's news
    for i, ts in enumerate(storm_timestamps):
        # Get the date (without time) for this timestamp
        date_only = ts.normalize()  # Strips time, keeps date
        
        # Find news for this date
        matching_news = storm_news[storm_news['date'] == date_only]
        
        if len(matching_news) > 0:
            aligned_articles[i] = matching_news['num_articles'].iloc[0]
        elif i > 0:
            # If no news for this day, forward fill from previous timestamp
            aligned_articles[i] = aligned_articles[i-1]
    
    return aligned_articles


def extract_satellite_data(file_path, fs):
    """Extract all 2D products from satellite file"""
    result = {'metadata': {}, 'images_2d': {}}
    
    with fs.open(file_path, 'rb') as file_handle:
        # Storm center position
        try:
            ds_storm = xr.open_dataset(file_handle, engine='h5netcdf', 
                                       group='overpass_storm_metadata', decode_timedelta=False)
            if 'storm_latitude' in ds_storm.variables:
                result['metadata']['sat_lat'] = safe_scalar(ds_storm['storm_latitude'].values)
            if 'storm_longitude' in ds_storm.variables:
                result['metadata']['sat_lon'] = safe_scalar(ds_storm['storm_longitude'].values)
            ds_storm.close()
        except:
            pass
        
        # IR imagery
        try:
            ds_ir = xr.open_dataset(file_handle, engine='h5netcdf', 
                                   group='infrared', decode_timedelta=False)
            var_name = get_variable(ds_ir, VAR_MAP['ir'])
            if var_name:
                img_2d = extract_2d_slice(ds_ir[var_name])
                # CRITICAL: Mask sentinel values BEFORE resizing
                img_2d = mask_sentinel_values(img_2d, var_type='ir')
                result['images_2d']['ir'] = resize_image(img_2d, TARGET_SIZE)
                
                # Calculate crop width
                if 'x' in ds_ir.variables and 'y' in ds_ir.variables:
                    x_vals, y_vals = ds_ir['x'].values, ds_ir['y'].values
                    x_extent = safe_scalar(np.abs(x_vals[-1] - x_vals[0]))
                    y_extent = safe_scalar(np.abs(y_vals[-1] - y_vals[0]))
                    result['metadata']['ir_crop_km'] = max(x_extent, y_extent)
            ds_ir.close()
        except:
            pass
        
        # GPROF products
        for subgroup in GPROF_SUBGROUPS:
            try:
                ds_gprof = xr.open_dataset(file_handle, engine='h5netcdf',
                                          group=f'GPROF/{subgroup}', decode_timedelta=False)
                
                # Calculate crop width
                if 'x' in ds_gprof.variables and 'y' in ds_gprof.variables:
                    x_vals, y_vals = ds_gprof['x'].values, ds_gprof['y'].values
                    x_extent = safe_scalar(np.abs(x_vals[-1] - x_vals[0]))
                    y_extent = safe_scalar(np.abs(y_vals[-1] - y_vals[0]))
                    result['metadata']['gprof_crop_km'] = max(x_extent, y_extent)
                
                # Extract all GPROF products
                for prod_name, var_names in VAR_MAP.items():
                    if prod_name == 'ir':
                        continue
                    var_name = get_variable(ds_gprof, var_names)
                    if var_name:
                        img_2d = extract_2d_slice(ds_gprof[var_name])
                        if img_2d.size > 0 and not np.all(np.isnan(img_2d)):
                            # CRITICAL: Mask sentinel values BEFORE resizing
                            img_2d = mask_sentinel_values(img_2d, var_type='gprof')
                            result['images_2d'][prod_name] = resize_image(img_2d, TARGET_SIZE)
                
                ds_gprof.close()
                break  # Found valid GPROF data, no need to check other subgroups
            except:
                continue
    
    return result


def process_storm_to_zarr(storm_id, storm_df, root_zarr, fs, news_df):
    """Process one storm and write to Zarr"""
    storm_df = storm_df.sort_values('Timestamp').reset_index(drop=True)
    n_times = len(storm_df)
    
    atcf_id = str(storm_df['atcf_id'].iloc[0])
    storm_name = str(storm_df['Storm_Name'].iloc[0]).strip("b'").strip("'")
    
    print(f"\n{'='*60}")
    print(f"Processing: {storm_name} ({atcf_id})")
    print(f"Timestamps: {n_times}")
    
    # Create storm group
    storm_group = root_zarr.create_group(atcf_id, overwrite=True)
    storm_group.attrs['storm_id'] = storm_id
    storm_group.attrs['storm_name'] = storm_name
    storm_group.attrs['basin'] = atcf_id[:2]
    storm_group.attrs['year'] = int(atcf_id[-4:])
    
    # Store timestamps
    timestamps = pd.to_datetime(storm_df['Timestamp']).values
    storm_group['timestamps'] = timestamps
    
    # Tabular data group
    tab_group = storm_group.create_group('tabular')
    tab_group['latitude'] = storm_df['Latitude'].values
    tab_group['longitude'] = storm_df['Longitude'].values
    tab_group['wind_max_knots'] = storm_df['Observed_Wind_Max_Knots'].fillna(np.nan).values
    tab_group['pressure_min_mb'] = storm_df['Observed_Pressure_Min_mb'].fillna(np.nan).values
    tab_group['storm_speed_knots'] = storm_df['Storm_Speed_Knots'].fillna(np.nan).values
    tab_group['storm_direction_deg'] = storm_df['Storm_Direction_Deg'].fillna(np.nan).values
    tab_group['era5_temp_2m'] = storm_df['ERA5_Temp_2m_Kelvin'].fillna(np.nan).values
    tab_group['era5_pressure_msl'] = storm_df['ERA5_Pressure_MSL_hPa'].fillna(np.nan).values
    tab_group['era5_wind_u'] = storm_df['ERA5_Wind_U_Component'].fillna(np.nan).values
    tab_group['era5_wind_v'] = storm_df['ERA5_Wind_V_Component'].fillna(np.nan).values
    tab_group['era5_position_error_km'] = storm_df['ERA5_Position_Error_km'].fillna(np.nan).values
    tab_group['rw_casualties'] = storm_df['RW_Casualty_Info'].fillna(0).values
    tab_group['rw_injured'] = storm_df['RW_Injured_Info'].fillna(0).values
    tab_group['rw_evacuated'] = storm_df['RW_Evacuated_Displaced'].fillna(0).values
    tab_group['rw_affected'] = storm_df['RW_Affected_Population'].fillna(0).values
    tab_group['rw_total_reports'] = storm_df['RW_Total_Reports'].fillna(0).values
    
    # NEW: News data group
    news_group = storm_group.create_group('news')
    aligned_articles = align_news_data(storm_df, news_df, storm_id)
    news_group['num_articles'] = aligned_articles
    print(f"  News: {np.sum(aligned_articles > 0)}/{n_times} timestamps with articles (total: {np.sum(aligned_articles)})")
    
    # Get S3 files
    year, basin, number = atcf_id[-4:], atcf_id[:2], atcf_id[2:4]
    s3_path = f"{S3_PATH}/{year}/{basin}/{number}"
    
    try:
        files = [f for f in fs.ls(s3_path) if f.endswith('.nc')]
    except:
        print(f"  ⚠️  No satellite files found")
        return
    
    # Parse timestamps and match to dataset timestamps
    sat_data = [(f, parse_timestamp_from_filename(os.path.basename(f))) for f in files]
    sat_data = [(f, t) for f, t in sat_data if t is not None]
    
    if not sat_data:
        print(f"  ⚠️  No valid timestamps in files")
        return
    
    tolerance = timedelta(hours=TIME_TOLERANCE_HOURS)
    matches = {}
    
    for idx, ds_time in enumerate(timestamps):
        ds_time_pd = pd.Timestamp(ds_time)
        best_file, best_diff = None, None
        
        for file_path, sat_time in sat_data:
            diff = abs(sat_time - ds_time_pd)
            if diff <= tolerance and (best_diff is None or diff < best_diff):
                best_file, best_diff = file_path, diff
        
        if best_file:
            matches[idx] = (best_file, best_diff.total_seconds() / 60, sat_time)
    
    print(f"  Matched: {len(matches)}/{n_times} timestamps")
    
    if len(matches) == 0:
        return
    
    # Pre-initialize ALL expected 2D arrays (prevents First Match Bug)
    img2d_group = storm_group.create_group('images_2d')
    for var_name in EXPECTED_2D_VARS:
        img2d_group.create(
            var_name,
            shape=(n_times, TARGET_SIZE[0], TARGET_SIZE[1]),
            dtype=np.float32,
            fill_value=np.nan,
            chunks=(1, TARGET_SIZE[0], TARGET_SIZE[1])
        )
    
    # Quality metadata arrays
    qual_group = storm_group.create_group('quality_meta')
    qual_group.create('time_diff_min', shape=(n_times,), dtype=np.float32, fill_value=np.nan)
    qual_group.create('center_dist_km', shape=(n_times,), dtype=np.float32, fill_value=np.nan)
    qual_group.create('ir_crop_km', shape=(n_times,), dtype=np.float32, fill_value=np.nan)
    qual_group.create('gprof_crop_km', shape=(n_times,), dtype=np.float32, fill_value=np.nan)
    
    # Extract and fill satellite data
    print(f"  Extracting satellite data...")
    for idx, (file_path, time_diff, sat_time) in matches.items():
        data = extract_satellite_data(file_path, fs)
        
        # Write images
        for var_name, img_data in data['images_2d'].items():
            if var_name in img2d_group:
                img2d_group[var_name][idx] = img_data
        
        # Write quality metadata
        qual_group['time_diff_min'][idx] = time_diff
        
        if 'sat_lat' in data['metadata'] and 'sat_lon' in data['metadata']:
            dist = haversine_distance(
                storm_df['Latitude'].iloc[idx], storm_df['Longitude'].iloc[idx],
                data['metadata']['sat_lat'], data['metadata']['sat_lon']
            )
            qual_group['center_dist_km'][idx] = dist
        
        if 'ir_crop_km' in data['metadata']:
            qual_group['ir_crop_km'][idx] = data['metadata']['ir_crop_km']
        if 'gprof_crop_km' in data['metadata']:
            qual_group['gprof_crop_km'][idx] = data['metadata']['gprof_crop_km']
    
    print(f"  ✅ Complete")


def build_dataset():
    """Build complete Zarr dataset for all storms"""
    print(f"\n{'='*60}")
    print(f"TC-PRIMED Dataset Builder with News Integration")
    print(f"{'='*60}")
    
    # Load and filter dataset
    print(f"\nLoading CSV...")
    df = pd.read_csv(INPUT_CSV)
    df['Storm_ID'] = df['Storm_ID'].apply(lambda x: str(x).strip("b'").strip("'"))
    df['atcf_id'] = df['atcf_id'].apply(lambda x: str(x).strip("b'").strip("'") if pd.notna(x) else None)
    df = df[df['atcf_id'].notna()].copy()
    unique_storms = df['Storm_ID'].unique()
    
    print(f"  Total storms: {len(unique_storms)}")
    print(f"  Total records: {len(df)}")
    
    # NEW: Load news data
    print(f"\nLoading news data from {NEWS_CSV}...")
    try:
        news_df = pd.read_csv(NEWS_CSV)
        news_df['Storm_ID'] = news_df['Storm_ID'].apply(lambda x: str(x).strip("b'").strip("'"))
        news_df['cyclone_name'] = news_df['cyclone_name'].apply(lambda x: str(x).strip("b'").strip("'"))
        print(f"  News records: {len(news_df)}")
        print(f"  Unique storms with news: {news_df['Storm_ID'].nunique()}")
        print(f"  Date range: {news_df['day'].min()} to {news_df['day'].max()}")
    except Exception as e:
        print(f"  ⚠️  Warning: Could not load news data: {e}")
        print(f"  Continuing without news data...")
        news_df = pd.DataFrame(columns=['Storm_ID', 'day', 'num_articles'])
    
    # Create Zarr and process storms
    print(f"\nCreating Zarr: {OUTPUT_ZARR}")
    root = zarr.open_group(OUTPUT_ZARR, mode='w')
    fs = s3fs.S3FileSystem(anon=True)
    
    for i, storm_id in enumerate(unique_storms, 1):
        storm_df = df[df['Storm_ID'] == storm_id]
        print(f"\n[{i}/{len(unique_storms)}]", end=" ")
        process_storm_to_zarr(storm_id, storm_df, root, fs, news_df)
    
    print(f"\n{'='*60}")
    print(f"✅ Dataset complete: {OUTPUT_ZARR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    build_dataset()