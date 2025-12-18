"""
TC-PRIMED Satellite Data Downloader (Optimized)
Fast byte-range extraction with filename-based timestamp parsing

USAGE: python tc_simple.py STORM_ID [TIME_TOLERANCE_HOURS]
EXAMPLE: python tc_simple.py 2022255N16256 1.5
"""

import pandas as pd
import numpy as np
import s3fs
import xarray as xr
from PIL import Image
import os
import sys
from datetime import timedelta
import re

# Configuration
INPUT_CSV = '../data/processed/ibtracs_era5_20251218_0052.csv'
OUTPUT_DIR = 'TC_PRIMED_Downloads'
S3_PATH = 'noaa-nesdis-tcprimed-pds/v01r01/final'
TARGET_SIZE = (224, 224)
DEFAULT_TIME_TOLERANCE_HOURS = 1.5

# Physical ranges for scientific normalization
PHYSICAL_RANGES = {
    'surface_precip': (0.0, 50.0),
    'convective_precip': (0.0, 40.0),
    'rain_water_path': (0.0, 5.0),
    'cloud_water_path': (0.0, 2.0),
    'ice_water_path': (0.0, 10.0),
    'precip_probability': (0.0, 100.0),
    'ir_temperature': (180.0, 320.0)
}

# Fallback configurations
GPROF_SUBGROUPS = ['S1', 'S2', 'S3']
VARIABLE_FALLBACKS = {
    'surface_precip': ['surfacePrecipitation', 'precip_rate'],
    'convective_precip': ['convectivePrecipitation'],
    'rain_water_path': ['rainWaterPath'],
    'cloud_water_path': ['cloudWaterPath'],
    'ice_water_path': ['iceWaterPath'],
    'precip_probability': ['probabilityOfPrecip'],
    'ir_brightness': ['IRWIN', 'brightness_temperature']
}


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in km between two points"""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def load_storm_data(storm_id):
    """Load storm data and timestamps from CSV"""
    df = pd.read_csv(INPUT_CSV)
    storm_id = str(storm_id).strip("b'").strip("'")
    df['Storm_ID'] = df['Storm_ID'].apply(lambda x: str(x).strip("b'").strip("'"))
    df['atcf_id'] = df['atcf_id'].apply(lambda x: str(x).strip("b'").strip("'") if pd.notna(x) else None)
    
    storm_data = df[df['Storm_ID'] == storm_id].copy()
    if len(storm_data) == 0:
        raise ValueError(f"Storm {storm_id} not found")
    
    storm_data['Timestamp'] = pd.to_datetime(storm_data['Timestamp'])
    first_row = storm_data.iloc[0]
    atcf = str(first_row['atcf_id'])
    
    positions = {row['Timestamp']: {'lat': row['Latitude'], 'lon': row['Longitude']} 
                 for _, row in storm_data.iterrows()}
    
    return {
        'name': first_row['Storm_Name'].strip("b'").strip("'"),
        'atcf': atcf,
        'year': atcf[-4:],
        'basin': atcf[:2],
        'number': atcf[2:4],
        'timestamps': sorted(storm_data['Timestamp'].tolist()),
        'positions': positions
    }


def parse_timestamp_from_filename(filename):
    """Extract timestamp from TC-PRIMED filename (fast, no network access)"""
    # Format: TCPRIMED_v01r01-final_STORMID_SENSOR_SAT_ORBIT_YYYYMMDDHHMMSS.nc
    match = re.search(r'_(\d{14})\.nc$', filename)
    if match:
        return pd.to_datetime(match.group(1), format='%Y%m%d%H%M%S')
    return None


def get_variable_with_fallback(ds, var_fallbacks):
    """Try multiple variable names until one is found"""
    for var_name in var_fallbacks:
        if var_name in ds.variables:
            return var_name
    return None


def extract_products_from_file(file_path, fs):
    """Extract all products from a single file using targeted byte-range requests"""
    result = {
        'metadata': {},
        'products': {'images': {}, 'gprof': {}}
    }
    
    # Open file ONCE and reuse handle for all groups
    file_handle = fs.open(file_path, 'rb')
    
    try:
        # Extract storm metadata
        try:
            ds_storm = xr.open_dataset(file_handle, engine='h5netcdf', 
                                      group='overpass_storm_metadata', decode_timedelta=False)
            if 'storm_latitude' in ds_storm.variables:
                result['metadata']['storm_center_lat'] = float(ds_storm['storm_latitude'].values[0])
            if 'storm_longitude' in ds_storm.variables:
                result['metadata']['storm_center_lon'] = float(ds_storm['storm_longitude'].values[0])
            ds_storm.close()
        except:
            pass
        
        # Extract IR imagery
        try:
            ds_ir = xr.open_dataset(file_handle, engine='h5netcdf', 
                                   group='infrared', decode_timedelta=False)
            var_name = get_variable_with_fallback(ds_ir, VARIABLE_FALLBACKS['ir_brightness'])
            
            if var_name:
                data = ds_ir[var_name]
                
                # Robust dimension handling using dimension names
                if 'time' in data.dims and data.sizes['time'] == 1:
                    data = data.isel(time=0)
                elif 'layer' in data.dims:
                    data = data.isel(layer=0)
                elif data.ndim > 2:
                    # Fallback: take first slice of first dimension
                    data = data.isel({data.dims[0]: 0})
                
                data = data.values
                if data.size > 0 and not np.all(np.isnan(data)):
                    result['products']['images']['ir'] = data
            
            ds_ir.close()
        except:
            pass
        
        # Extract GPROF products (try subgroups in order)
        # Get all possible GPROF variable names for fast intersection check
        all_gprof_vars = set()
        for var_list in VARIABLE_FALLBACKS.values():
            if var_list != VARIABLE_FALLBACKS['ir_brightness']:
                all_gprof_vars.update(var_list)
        
        for subgroup in GPROF_SUBGROUPS:
            try:
                ds_gprof = xr.open_dataset(file_handle, engine='h5netcdf', 
                                          group=f'GPROF/{subgroup}', decode_timedelta=False)
                
                # Fast metadata-only check: does this subgroup have any GPROF variables?
                if set(ds_gprof.data_vars).intersection(all_gprof_vars):
                    # Extract GPROF products
                    for product_name, var_fallbacks in VARIABLE_FALLBACKS.items():
                        if product_name == 'ir_brightness':
                            continue
                        
                        var_name = get_variable_with_fallback(ds_gprof, var_fallbacks)
                        if var_name:
                            data = ds_gprof[var_name]
                            
                            # Robust dimension handling
                            if 'layer' in data.dims:
                                # Take surface layer (index 0)
                                data = data.isel(layer=0)
                            elif 'time' in data.dims and data.sizes['time'] == 1:
                                data = data.isel(time=0)
                            elif data.ndim == 3:
                                # Fallback: assume first dimension is layer/time
                                data = data.isel({data.dims[0]: 0})
                            
                            data = data.values
                            if data.size > 0 and not np.all(np.isnan(data)):
                                result['products']['gprof'][product_name] = data
                    
                    ds_gprof.close()
                    break  # Found valid subgroup, stop searching
                
                ds_gprof.close()
            except:
                continue
    
    finally:
        file_handle.close()
    
    return result


def normalize_and_save(data, output_path, product_type):
    """Normalize data using physical ranges and save as PNG"""
    if product_type in PHYSICAL_RANGES:
        vmin, vmax = PHYSICAL_RANGES[product_type]
    else:
        vmin, vmax = np.nanmin(data), np.nanmax(data)
    
    if vmax > vmin:
        data_norm = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    else:
        data_norm = np.zeros_like(data)
    
    data_norm = np.nan_to_num(data_norm, nan=0.0)
    
    img = Image.fromarray((data_norm * 255).astype(np.uint8), mode='L')
    img_resized = img.resize(TARGET_SIZE, Image.LANCZOS)
    img_resized.save(output_path)


def download_storm_images(storm_id, time_tolerance_hours=DEFAULT_TIME_TOLERANCE_HOURS):
    """Main download function"""
    print(f"\n🌀 Loading storm data...")
    info = load_storm_data(storm_id)
    print(f"   Storm: {info['name']} ({info['atcf']})")
    print(f"   Dataset timestamps: {len(info['timestamps'])}")
    
    # Connect to S3
    print(f"\n📡 Connecting to S3...")
    fs = s3fs.S3FileSystem(anon=True)
    s3_path = f"{S3_PATH}/{info['year']}/{info['basin']}/{info['number']}"
    files = [f for f in fs.ls(s3_path) if f.endswith('.nc')]
    print(f"   Found {len(files)} satellite files")
    
    # Parse timestamps from filenames (INSTANT - no network access!)
    print(f"\n📥 Parsing timestamps from filenames...")
    satellite_data = []
    for file_path in files:
        timestamp = parse_timestamp_from_filename(os.path.basename(file_path))
        if timestamp:
            satellite_data.append({'file_path': file_path, 'timestamp': timestamp})
    
    print(f"   Valid files: {len(satellite_data)}")
    
    # Match closest satellite to each dataset timestamp
    print(f"\n🔍 Matching timestamps (±{time_tolerance_hours}h)...")
    tolerance = timedelta(hours=time_tolerance_hours)
    matches = []
    
    for ds_timestamp in info['timestamps']:
        best = None
        best_diff = None
        for sat_data in satellite_data:
            diff = abs(sat_data['timestamp'] - ds_timestamp)
            if diff <= tolerance and (best is None or diff < best_diff):
                best = sat_data
                best_diff = diff
        if best:
            matches.append({
                'dataset_timestamp': ds_timestamp,
                'satellite_timestamp': best['timestamp'],
                'time_diff': best_diff,
                'file_path': best['file_path']
            })
    
    print(f"   Matched: {len(matches)} timestamps")
    
    # Download products (only for matched files!)
    output_dir = os.path.join(OUTPUT_DIR, f"{info['name']}_{info['atcf']}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📥 Downloading products...\n")
    results = []
    
    for i, match in enumerate(matches, 1):
        ds_timestamp = match['dataset_timestamp']
        time_diff_min = match['time_diff'].total_seconds() / 60
        
        # Extract products (byte-range requests only for this file)
        data = extract_products_from_file(match['file_path'], fs)
        
        if not data['products']['images'] and not data['products']['gprof']:
            continue
        
        # Calculate distance
        dataset_pos = info['positions'][ds_timestamp]
        distance_km = None
        if 'storm_center_lat' in data['metadata'] and 'storm_center_lon' in data['metadata']:
            distance_km = haversine_distance(
                dataset_pos['lat'], dataset_pos['lon'],
                data['metadata']['storm_center_lat'], 
                data['metadata']['storm_center_lon']
            )
        
        # Save products
        timestamp_str = ds_timestamp.strftime('%Y%m%d_%H%M%S')
        saved = []
        
        # IR imagery
        if 'ir' in data['products']['images']:
            path = os.path.join(output_dir, f"{info['name']}_{timestamp_str}_ir.png")
            normalize_and_save(data['products']['images']['ir'], path, 'ir_temperature')
            saved.append('IR')
        
        # GPROF products
        for prod_name, prod_data in data['products']['gprof'].items():
            path = os.path.join(output_dir, f"{info['name']}_{timestamp_str}_gprof_{prod_name}.png")
            normalize_and_save(prod_data, path, prod_name)
            saved.append('GPROF')
        
        # Log progress
        products_str = '+'.join(saved[:3])
        if len(saved) > 3:
            products_str += f"+{len(saved)-3}more"
        dist_str = f", Δ{distance_km:.1f}km" if distance_km else ""
        print(f"   ✅ [{i}/{len(matches)}] {ds_timestamp.strftime('%Y-%m-%d %H:%M')} "
              f"({products_str}, Δ{time_diff_min:.1f}min{dist_str})")
        
        # Store metadata
        result = {
            'dataset_timestamp': ds_timestamp,
            'satellite_timestamp': match['satellite_timestamp'],
            'time_diff_minutes': time_diff_min,
            'dataset_lat': dataset_pos['lat'],
            'dataset_lon': dataset_pos['lon'],
            'satellite_storm_center_lat': data['metadata'].get('storm_center_lat'),
            'satellite_storm_center_lon': data['metadata'].get('storm_center_lon'),
            'center_distance_km': distance_km,
            'infrared_available': 'ir' in data['products']['images'],
            'gprof_products_count': len(data['products']['gprof']),
            'filename': os.path.basename(match['file_path'])
        }
        
        # Add product paths
        if 'ir' in data['products']['images']:
            result['image_path_ir'] = os.path.join(output_dir, f"{info['name']}_{timestamp_str}_ir.png")
        for prod_name in data['products']['gprof'].keys():
            result[f'gprof_path_{prod_name}'] = os.path.join(
                output_dir, f"{info['name']}_{timestamp_str}_gprof_{prod_name}.png")
        
        results.append(result)
    
    # Save metadata CSV
    if results:
        df = pd.DataFrame(results).sort_values('dataset_timestamp')
        df.to_csv(os.path.join(output_dir, 'matched_images.csv'), index=False)
    
    print(f"\n{'='*60}")
    print(f"✅ Saved: {len(results)}/{len(matches)} timestamps")
    print(f"📁 Output: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tc_simple.py STORM_ID [TIME_TOLERANCE_HOURS]")
        print("Example: python tc_simple.py 2022255N16256 1.5")
        sys.exit(1)
    
    storm_id = sys.argv[1]
    time_tolerance = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_TIME_TOLERANCE_HOURS
    
    download_storm_images(storm_id, time_tolerance)