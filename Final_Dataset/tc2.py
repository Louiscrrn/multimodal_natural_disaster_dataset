"""
TC-PRIMED Satellite Data Downloader (Optimized)
Fast extraction with Zarr storage for full 3D data + PNG visualizations

Features:
- Filename-based timestamp parsing (instant, no network overhead)
- Single file handle per file (minimal S3 connections)
- Full 3D data stored in Zarr format
- Log-transformed precipitation for ML
- 2D PNG slices for visualization
- Pixel resolution extraction

USAGE: python tc_simple.py STORM_ID [TIME_TOLERANCE_HOURS]
EXAMPLE: python tc_simple.py 2022255N16256 1.5

Requirements: pip install xarray h5netcdf zarr
"""

import pandas as pd
import numpy as np
import s3fs
import xarray as xr
import zarr
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
    """Extract all products from a single file with full dimensionality"""
    result = {
        'metadata': {},
        'products': {'images': {}, 'gprof': {}},
        'coordinates': {},
        'dimensions': {}
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
        
        # Extract IR imagery with full dimensions
        try:
            ds_ir = xr.open_dataset(file_handle, engine='h5netcdf', 
                                   group='infrared', decode_timedelta=False)
            var_name = get_variable_with_fallback(ds_ir, VARIABLE_FALLBACKS['ir_brightness'])
            
            if var_name:
                data_full = ds_ir[var_name]
                
                # Extract resolution if available
                if 'x' in ds_ir.variables and 'y' in ds_ir.variables:
                    x_coords = ds_ir['x'].values
                    y_coords = ds_ir['y'].values
                    if len(x_coords) > 1 and len(y_coords) > 1:
                        x_res = np.abs(np.median(np.diff(x_coords)))
                        y_res = np.abs(np.median(np.diff(y_coords)))
                        result['metadata']['ir_resolution_km'] = float((x_res + y_res) / 2)
                
                # Store full dimensional data
                result['products']['images']['ir_full'] = data_full.values
                result['dimensions']['ir'] = list(data_full.dims)
                
                # Also extract 2D slice for PNG visualization
                data_2d = data_full
                if 'time' in data_2d.dims and data_2d.sizes['time'] == 1:
                    data_2d = data_2d.isel(time=0)
                elif 'layer' in data_2d.dims:
                    data_2d = data_2d.isel(layer=0)
                elif data_2d.ndim > 2:
                    data_2d = data_2d.isel({data_2d.dims[0]: 0})
                
                data_2d = data_2d.values
                if data_2d.size > 0 and not np.all(np.isnan(data_2d)):
                    result['products']['images']['ir_2d'] = data_2d
            
            ds_ir.close()
        except:
            pass
        
        # Extract GPROF products with full dimensions
        all_gprof_vars = set()
        for var_list in VARIABLE_FALLBACKS.values():
            if var_list != VARIABLE_FALLBACKS['ir_brightness']:
                all_gprof_vars.update(var_list)
        
        for subgroup in GPROF_SUBGROUPS:
            try:
                ds_gprof = xr.open_dataset(file_handle, engine='h5netcdf', 
                                          group=f'GPROF/{subgroup}', decode_timedelta=False)
                
                # Fast metadata-only check
                if set(ds_gprof.data_vars).intersection(all_gprof_vars):
                    # Extract spatial resolution
                    if 'x' in ds_gprof.variables and 'y' in ds_gprof.variables:
                        x_coords = ds_gprof['x'].values
                        y_coords = ds_gprof['y'].values
                        if len(x_coords) > 1 and len(y_coords) > 1:
                            x_res = np.abs(np.median(np.diff(x_coords)))
                            y_res = np.abs(np.median(np.diff(y_coords)))
                            result['metadata']['gprof_resolution_km'] = float((x_res + y_res) / 2)
                    
                    # Store coordinates
                    if 'latitude' in ds_gprof.variables:
                        result['coordinates']['lat'] = ds_gprof['latitude'].values
                    if 'longitude' in ds_gprof.variables:
                        result['coordinates']['lon'] = ds_gprof['longitude'].values
                    
                    # Extract GPROF products with full dimensions
                    for product_name, var_fallbacks in VARIABLE_FALLBACKS.items():
                        if product_name == 'ir_brightness':
                            continue
                        
                        var_name = get_variable_with_fallback(ds_gprof, var_fallbacks)
                        if var_name:
                            data_full = ds_gprof[var_name]
                            
                            # Store full dimensional data
                            full_data = data_full.values
                            if full_data.size > 0:
                                result['products']['gprof'][f'{product_name}_full'] = full_data
                                result['dimensions'][product_name] = list(data_full.dims)
                            
                            # Extract 2D slice for PNG
                            data_2d = data_full
                            if 'layer' in data_2d.dims:
                                data_2d = data_2d.isel(layer=0)
                            elif 'time' in data_2d.dims and data_2d.sizes['time'] == 1:
                                data_2d = data_2d.isel(time=0)
                            elif data_2d.ndim == 3:
                                data_2d = data_2d.isel({data_2d.dims[0]: 0})
                            
                            data_2d = data_2d.values
                            if data_2d.size > 0 and not np.all(np.isnan(data_2d)):
                                result['products']['gprof'][f'{product_name}_2d'] = data_2d
                    
                    ds_gprof.close()
                    break
                
                ds_gprof.close()
            except:
                continue
    
    finally:
        file_handle.close()
    
    return result


def normalize_and_save_png(data, output_path, product_type):
    """Normalize data using physical ranges (with log transform for precip) and save as PNG"""
    # Apply log transform for precipitation data (heavy-tailed distribution)
    is_precip = 'precip' in product_type or 'rain' in product_type or 'water' in product_type
    if is_precip:
        data = np.log1p(data)  # log(1 + x) to handle zeros
        # Adjust ranges for log-transformed data
        if product_type == 'surface_precip':
            vmin, vmax = 0.0, np.log1p(50.0)
        elif product_type == 'convective_precip':
            vmin, vmax = 0.0, np.log1p(40.0)
        else:
            vmin, vmax = np.nanmin(data), np.nanmax(data)
    elif product_type in PHYSICAL_RANGES:
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


def save_to_zarr(data_dict, zarr_path, metadata, coordinates, dimensions):
    """Save full dimensional data to Zarr format"""
    # Create Zarr group (modern API)
    root = zarr.open_group(zarr_path, mode='w')
    
    # Save metadata
    root.attrs.update(metadata)
    
    # Save coordinates
    if coordinates:
        coords_group = root.create_group('coordinates')
        for coord_name, coord_data in coordinates.items():
            coords_group[coord_name] = coord_data
    
    # Save IR data
    if 'images' in data_dict:
        img_group = root.create_group('images')
        for var_name, var_data in data_dict['images'].items():
            if '_full' in var_name:
                clean_name = var_name.replace('_full', '')
                img_group[clean_name] = var_data
                # Store dimensions
                if clean_name in dimensions:
                    img_group[clean_name].attrs['dimensions'] = dimensions[clean_name]
    
    # Save GPROF data
    if 'gprof' in data_dict:
        gprof_group = root.create_group('gprof')
        for var_name, var_data in data_dict['gprof'].items():
            if '_full' in var_name:
                clean_name = var_name.replace('_full', '')
                gprof_group[clean_name] = var_data
                # Store dimensions
                if clean_name in dimensions:
                    gprof_group[clean_name].attrs['dimensions'] = dimensions[clean_name]


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
    zarr_dir = os.path.join(output_dir, 'zarr')
    png_dir = os.path.join(output_dir, 'png')
    os.makedirs(zarr_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)
    
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
        
        # Save files
        timestamp_str = ds_timestamp.strftime('%Y%m%d_%H%M%S')
        saved = []
        
        # Save full data to Zarr
        zarr_path = os.path.join(zarr_dir, f"{info['name']}_{timestamp_str}.zarr")
        save_to_zarr(data['products'], zarr_path, data['metadata'], 
                    data['coordinates'], data['dimensions'])
        saved.append('ZARR')
        
        # Save 2D slices as PNG for visualization
        if 'ir_2d' in data['products']['images']:
            path = os.path.join(png_dir, f"{info['name']}_{timestamp_str}_ir.png")
            normalize_and_save_png(data['products']['images']['ir_2d'], path, 'ir_temperature')
            saved.append('IR')
        
        # GPROF products (PNG with log transform for precip)
        for var_name, var_data in data['products']['gprof'].items():
            if '_2d' in var_name:
                prod_name = var_name.replace('_2d', '')
                path = os.path.join(png_dir, f"{info['name']}_{timestamp_str}_gprof_{prod_name}.png")
                normalize_and_save_png(var_data, path, prod_name)
                saved.append('GPROF')
        
        # Log progress
        products_str = '+'.join(list(set(saved))[:4])
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
            'ir_resolution_km': data['metadata'].get('ir_resolution_km'),
            'gprof_resolution_km': data['metadata'].get('gprof_resolution_km'),
            'infrared_available': 'ir_full' in data['products']['images'],
            'gprof_products_count': len([k for k in data['products']['gprof'].keys() if '_full' in k]),
            'zarr_path': zarr_path,
            'filename': os.path.basename(match['file_path'])
        }
        
        # Add PNG paths
        if 'ir_2d' in data['products']['images']:
            result['png_path_ir'] = os.path.join(png_dir, f"{info['name']}_{timestamp_str}_ir.png")
        
        for var_name in data['products']['gprof'].keys():
            if '_2d' in var_name:
                prod_name = var_name.replace('_2d', '')
                result[f'png_path_gprof_{prod_name}'] = os.path.join(
                    png_dir, f"{info['name']}_{timestamp_str}_gprof_{prod_name}.png")
        
        results.append(result)
    
    # Save metadata CSV
    if results:
        df = pd.DataFrame(results).sort_values('dataset_timestamp')
        df.to_csv(os.path.join(output_dir, 'matched_images.csv'), index=False)
    
    print(f"\n{'='*60}")
    print(f"✅ Saved: {len(results)}/{len(matches)} timestamps")
    print(f"📁 Output:")
    print(f"   - Full data (Zarr): {zarr_dir}")
    print(f"   - Visualizations (PNG): {png_dir}")
    print(f"   - Metadata CSV: {output_dir}/matched_images.csv")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("TC-PRIMED Satellite Data Downloader")
        print("\nUsage: python tc_simple.py STORM_ID [TIME_TOLERANCE_HOURS]")
        print("\nExamples:")
        print("  python tc_simple.py 2022255N16256")
        print("  python tc_simple.py 2022255N16256 3.0")
        print("\nOutput:")
        print("  - zarr/ : Full 3D data in Zarr format")
        print("  - png/  : 2D visualizations (log-transformed precip)")
        print("  - matched_images.csv : Metadata with resolutions")
        print("\nRequirements: pip install xarray h5netcdf zarr")
        sys.exit(1)
    
    storm_id = sys.argv[1]
    time_tolerance = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_TIME_TOLERANCE_HOURS
    
    download_storm_images(storm_id, time_tolerance)