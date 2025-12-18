import pandas as pd
import numpy as np
import s3fs
import xarray as xr
import zarr
import os
import re
import ast
from datetime import timedelta
from scipy.ndimage import zoom
import warnings
from zarr.errors import UnstableSpecificationWarning

warnings.filterwarnings(
    "ignore",
    category=UnstableSpecificationWarning
)

# Configuration
INPUT_CSV = '/data/processed/ibtracs_era5_20251218_1715_reliefweb.csv'
NEWS_CSV = '/data/processed/cyclones_mentions_gdelt_3h_2022_2023.csv'
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

EXPECTED_2D_VARS = list(VAR_MAP.keys())

def safe_scalar(value):
    if isinstance(value, np.ndarray):
        return float(value.flat[0] if value.size > 0 else np.nan)
    return float(value)


def safe_parse_list(value):
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, list) else []
    except:
        return []


def mask_sentinel_values(data, var_type='gprof'):
    """Mask sentinel/fill values as NaN before processing"""
    data = data.astype(np.float32)
    
    if var_type == 'ir':
        data[data == 0.0] = np.nan
        data[data < 100.0] = np.nan 
    else:  # GPROF
        data[data == -9999.0] = np.nan
        data[data < -1000.0] = np.nan 
    
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
    for var in var_names:
        if var in ds.variables:
            return var
    return None


def extract_2d_slice(data):
    while data.ndim > 2:
        # Remove first dimension if it's singleton or time
        if data.shape[0] == 1 or (hasattr(data, 'dims') and data.dims[0] in ['time', 'layer']):
            data = data[0] if isinstance(data, np.ndarray) else data.isel({data.dims[0]: 0})
        else:
            break
    return data.values if hasattr(data, 'values') else data


def align_news_data(storm_df, news_df, storm_id):
    """Align daily news data with 3-hourly storm timestamps"""

    storm_news = news_df[news_df['Storm_ID'] == storm_id].copy()
    
    n_times = len(storm_df)
    
    if len(storm_news) == 0:
        return (
            np.zeros(n_times, dtype=np.int32),
            np.array([""] * n_times, dtype=object),
            np.array([""] * n_times, dtype=object)
        )
    
    # Convert day column to datetime
    storm_news['date'] = pd.to_datetime(storm_news['day'], format='%Y%m%d')
    storm_news = storm_news.sort_values('date').reset_index(drop=True)
    
    daily_lookup = {}
    for _, row in storm_news.iterrows():
        date_only = row['date'].normalize()
        
        # Parse URL and source lists (join with ",")
        urls = safe_parse_list(row.get('url', []))
        sources = safe_parse_list(row.get('source', []))
        
        urls_str = ", ".join(urls) if urls else ""
        sources_str = ", ".join(sources) if sources else ""
        
        daily_lookup[date_only] = (
            int(row['num_articles']),
            urls_str,
            sources_str
        )
    
    storm_timestamps = pd.to_datetime(storm_df['Timestamp'])
    aligned_articles = np.zeros(n_times, dtype=np.int32)
    aligned_urls = np.array([""] * n_times, dtype=object)
    aligned_sources = np.array([""] * n_times, dtype=object)
    
    # for each timestamp find the corresponding day
    for i, ts in enumerate(storm_timestamps):
        date_only = ts.normalize()
        
        if date_only in daily_lookup:
            num_art, urls_str, sources_str = daily_lookup[date_only]
            aligned_articles[i] = num_art
            aligned_urls[i] = urls_str
            aligned_sources[i] = sources_str
        elif i > 0:
            # Forward fill from previous timestamp
            aligned_articles[i] = aligned_articles[i-1]
            aligned_urls[i] = aligned_urls[i-1]
            aligned_sources[i] = aligned_sources[i-1]
    
    return aligned_articles, aligned_urls, aligned_sources


def extract_satellite_data(file_path, fs):
    """Extract all 2D products from satellite file"""
    result = {'metadata': {}, 'images': {}}
    
    with fs.open(file_path, 'rb') as file_handle:
        try:
            ds_storm = xr.open_dataset(file_handle, engine='h5netcdf', group='overpass_storm_metadata', decode_timedelta=False)
            if 'storm_latitude' in ds_storm.variables:
                result['metadata']['sat_lat'] = safe_scalar(ds_storm['storm_latitude'].values)
            if 'storm_longitude' in ds_storm.variables:
                result['metadata']['sat_lon'] = safe_scalar(ds_storm['storm_longitude'].values)
            ds_storm.close()
        except:
            pass
        
        # IR images
        try:
            ds_ir = xr.open_dataset(file_handle, engine='h5netcdf', group='infrared', decode_timedelta=False)
            var_name = get_variable(ds_ir, VAR_MAP['ir'])
            if var_name:
                img_2d = extract_2d_slice(ds_ir[var_name])
                # Mask sentinel values before resizing
                img_2d = mask_sentinel_values(img_2d, var_type='ir')
                result['images']['ir'] = resize_image(img_2d, TARGET_SIZE)
                
                # Calculate crop width
                if 'x' in ds_ir.variables and 'y' in ds_ir.variables:
                    x_vals, y_vals = ds_ir['x'].values, ds_ir['y'].values
                    x_extent = safe_scalar(np.abs(x_vals[-1] - x_vals[0]))
                    y_extent = safe_scalar(np.abs(y_vals[-1] - y_vals[0]))
                    result['metadata']['ir_crop_km'] = max(x_extent, y_extent)
            ds_ir.close()
        except:
            pass
        
        # GPROF images
        for subgroup in GPROF_SUBGROUPS:
            try:
                ds_gprof = xr.open_dataset(file_handle, engine='h5netcdf', group=f'GPROF/{subgroup}', decode_timedelta=False)
                
                # Calculate crop width
                if 'x' in ds_gprof.variables and 'y' in ds_gprof.variables:
                    x_vals, y_vals = ds_gprof['x'].values, ds_gprof['y'].values
                    x_extent = safe_scalar(np.abs(x_vals[-1] - x_vals[0]))
                    y_extent = safe_scalar(np.abs(y_vals[-1] - y_vals[0]))
                    result['metadata']['gprof_crop_km'] = max(x_extent, y_extent)
                
                for prod_name, var_names in VAR_MAP.items():
                    if prod_name == 'ir':
                        continue
                    var_name = get_variable(ds_gprof, var_names)
                    if var_name:
                        img_2d = extract_2d_slice(ds_gprof[var_name])
                        if img_2d.size > 0 and not np.all(np.isnan(img_2d)):
                            # Mask sentinel values before resizing
                            img_2d = mask_sentinel_values(img_2d, var_type='gprof')
                            result['images'][prod_name] = resize_image(img_2d, TARGET_SIZE)
                
                ds_gprof.close()
                break
            except:
                continue
    
    return result


def process_storm_to_zarr(storm_id, storm_df, root_zarr, fs, news_df):
    """Process one storm and write to Zarr"""
    storm_df = storm_df.sort_values('Timestamp').reset_index(drop=True)
    n_times = len(storm_df)
    
    atcf_id = str(storm_df['atcf_id'].iloc[0])
    storm_name = str(storm_df['Storm_Name'].iloc[0]).strip("b'").strip("'")
    
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
    
    # News data group with string arrays
    news_group = storm_group.create_group('news')
    aligned_articles, aligned_urls, aligned_sources = align_news_data(storm_df, news_df, storm_id)
    
    news_group['num_articles'] = aligned_articles
    
    MAX_LEN = 2048  # fixed size easier for Zarr but with high value to make truncation unlikely

    urls_bytes = np.array(
        [s.encode("utf-8")[:MAX_LEN] for s in aligned_urls],
        dtype=f"S{MAX_LEN}"
    )

    sources_bytes = np.array(
        [s.encode("utf-8")[:MAX_LEN] for s in aligned_sources],
        dtype=f"S{MAX_LEN}"
    )

    news_group.create_array(
        'urls',
        data=urls_bytes
    )

    news_group.create_array(
        'sources',
        data=sources_bytes
    )
    
    # Add description
    news_group.attrs['description'] = ("URLs and sources are strings separated with commas (Use .split(','))")
    
    # Get S3 files
    year, basin, number = atcf_id[-4:], atcf_id[:2], atcf_id[2:4]
    s3_path = f"{S3_PATH}/{year}/{basin}/{number}"
    files = [f for f in fs.ls(s3_path) if f.endswith('.nc')]
    
    # Parse timestamps and match to dataset timestamps
    sat_data = [(f, parse_timestamp_from_filename(os.path.basename(f))) for f in files]
    sat_data = [(f, t) for f, t in sat_data if t is not None]
    
    if not sat_data:
        print(f"No valid timestamps in files")
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
    
    if len(matches) == 0:
        return
    
    # Pre-initialize all expected 2D arrays
    img2d_group = storm_group.create_group('images')
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
    for idx, (file_path, time_diff, sat_time) in matches.items():
        data = extract_satellite_data(file_path, fs)
        
        for var_name, img_data in data['images'].items():
            if var_name in img2d_group:
                img2d_group[var_name][idx] = img_data
        
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


def build_dataset():
    """Build complete Zarr dataset for all storms"""
    
    df = pd.read_csv(INPUT_CSV)
    df['Storm_ID'] = df['Storm_ID'].apply(lambda x: str(x).strip("b'").strip("'"))
    df['atcf_id'] = df['atcf_id'].apply(lambda x: str(x).strip("b'").strip("'") if pd.notna(x) else None)
    df = df[df['atcf_id'].notna()].copy()
    unique_storms = df['Storm_ID'].unique()

    news_df = pd.read_csv(NEWS_CSV)
    news_df['Storm_ID'] = news_df['Storm_ID'].apply(lambda x: str(x).strip("b'").strip("'"))
    news_df['cyclone_name'] = news_df['cyclone_name'].apply(lambda x: str(x).strip("b'").strip("'"))
    
    # Create Zarr and process storms
    root = zarr.open_group(OUTPUT_ZARR, mode='w')
    fs = s3fs.S3FileSystem(anon=True)
    
    for i, storm_id in enumerate(unique_storms, 1):
        storm_df = df[df['Storm_ID'] == storm_id]
        process_storm_to_zarr(storm_id, storm_df, root, fs, news_df)


if __name__ == "__main__":
    build_dataset()