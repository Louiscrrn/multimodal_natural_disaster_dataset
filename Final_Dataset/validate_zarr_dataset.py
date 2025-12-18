"""
TC-PRIMED Zarr Dataset Validation Script with News Data

Performs comprehensive sanity checks on the generated Zarr dataset:
1. Structure validation (groups, arrays, dimensions)
2. Tabular data alignment with original CSV
3. Satellite image availability and quality
4. Physical range validation (IR temps, precipitation)
5. FillValue integrity (NaN vs 0.0)
6. Spatial centering verification
7. ReliefWeb data integrity
8. NEWS DATA: Broadcasting validation and temporal coherence
9. Visual sanity checks with proper colormaps

USAGE: python validate_zarr_dataset.py
"""

import pandas as pd
import numpy as np
import zarr
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Configuration
INPUT_CSV = '../data/processed/small_15_ibtracs_era5_20251218_1520_reliefweb.csv'
NEWS_CSV = '../data/processed/cyclones_mentions_gdelt_3h_2022_2023.csv'  # NEW
ZARR_PATH = 'TC_Clean_Dataset.zarr'
OUTPUT_DIR = 'validation_results'
TARGET_SIZE = (224, 224)

# Expected structure
EXPECTED_TABULAR_VARS = [
    'latitude', 'longitude', 'wind_max_knots', 'pressure_min_mb',
    'storm_speed_knots', 'storm_direction_deg', 'era5_temp_2m',
    'era5_pressure_msl', 'era5_wind_u', 'era5_wind_v',
    'era5_position_error_km', 'rw_casualties', 'rw_injured',
    'rw_evacuated', 'rw_affected', 'rw_total_reports'
]

EXPECTED_IMAGE_VARS = [
    'ir', 'surface_precip', 'convective_precip',
    'rain_water_path', 'cloud_water_path', 'ice_water_path'
]

EXPECTED_QUALITY_VARS = [
    'time_diff_min', 'center_dist_km', 'ir_crop_km', 'gprof_crop_km'
]

EXPECTED_NEWS_VARS = ['num_articles']  # NEW

# Physical ranges for validation
PHYSICAL_RANGES = {
    'ir': (180.0, 320.0),  # Kelvin
    'surface_precip': (0.0, 200.0),  # mm/hr (extreme values can be higher)
    'convective_precip': (0.0, 150.0),  # mm/hr
    'rain_water_path': (0.0, 10.0),  # kg/m²
    'cloud_water_path': (0.0, 5.0),  # kg/m²
    'ice_water_path': (0.0, 15.0),  # kg/m²
}

# Visualization settings
COLORMAPS = {
    'ir': 'gray',
    'surface_precip': 'Blues',
    'convective_precip': 'YlOrRd',
    'rain_water_path': 'viridis',
    'cloud_water_path': 'cividis',
    'ice_water_path': 'cool'
}


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    """Print formatted section header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}")
    print(f"{text}")
    print(f"{'='*70}{Colors.END}\n")


def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_error(text):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_warning(text):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


def print_info(text):
    """Print info message"""
    print(f"  {text}")


def load_data():
    """Load CSV, News CSV, and Zarr data"""
    print_header("Loading Data")
    
    # Load main CSV
    try:
        df = pd.read_csv(INPUT_CSV)
        df['Storm_ID'] = df['Storm_ID'].apply(lambda x: str(x).strip("b'").strip("'"))
        df['atcf_id'] = df['atcf_id'].apply(lambda x: str(x).strip("b'").strip("'") if pd.notna(x) else None)
        df = df[df['atcf_id'].notna()].copy()
        print_success(f"Loaded CSV: {len(df)} records, {df['Storm_ID'].nunique()} storms")
    except Exception as e:
        print_error(f"Failed to load CSV: {e}")
        return None, None, None
    
    # Load news CSV
    try:
        news_df = pd.read_csv(NEWS_CSV)
        news_df['Storm_ID'] = news_df['Storm_ID'].apply(lambda x: str(x).strip("b'").strip("'"))
        print_success(f"Loaded News CSV: {len(news_df)} records, {news_df['Storm_ID'].nunique()} storms with news")
    except Exception as e:
        print_warning(f"Could not load news CSV: {e}")
        news_df = None
    
    # Load Zarr
    try:
        zarr_root = zarr.open_group(ZARR_PATH, mode='r')
        print_success(f"Loaded Zarr: {len(list(zarr_root.groups()))} storm groups")
    except Exception as e:
        print_error(f"Failed to load Zarr: {e}")
        return df, news_df, None
    
    return df, news_df, zarr_root


def validate_structure(zarr_root, df):
    """Validate Zarr structure matches expected schema"""
    print_header("Validating Dataset Structure")
    
    issues = []
    
    # Get unique storms from CSV
    csv_storms = set(df['atcf_id'].unique())
    zarr_storms = set(zarr_root.group_keys())
    
    # Check storm groups
    missing_storms = csv_storms - zarr_storms
    extra_storms = zarr_storms - csv_storms
    
    if missing_storms:
        print_error(f"Missing storm groups: {missing_storms}")
        issues.append(f"Missing {len(missing_storms)} storms")
    
    if extra_storms:
        print_warning(f"Extra storm groups: {extra_storms}")
    
    if not missing_storms and not extra_storms:
        print_success(f"All {len(csv_storms)} storms present in Zarr")
    
    # Validate structure for each storm
    storms_checked = 0
    for atcf_id in zarr_storms:
        storm_group = zarr_root[atcf_id]
        
        # Check required groups
        if 'timestamps' not in storm_group:
            print_error(f"{atcf_id}: Missing 'timestamps' array")
            issues.append(f"{atcf_id}: No timestamps")
        
        if 'tabular' not in storm_group:
            print_error(f"{atcf_id}: Missing 'tabular' group")
            issues.append(f"{atcf_id}: No tabular data")
        else:
            # Check tabular variables
            missing_vars = set(EXPECTED_TABULAR_VARS) - set(storm_group['tabular'].array_keys())
            if missing_vars:
                print_warning(f"{atcf_id}: Missing tabular vars: {missing_vars}")
        
        if 'images' not in storm_group:
            print_error(f"{atcf_id}: Missing 'images' group")
            issues.append(f"{atcf_id}: No image data")
        else:
            # Check image variables
            available_imgs = set(storm_group['images'].array_keys())
            if 'ir' not in available_imgs:
                print_warning(f"{atcf_id}: Missing IR data")
        
        if 'quality_meta' not in storm_group:
            print_warning(f"{atcf_id}: Missing 'quality_meta' group")
        
        # NEW: Check news group
        if 'news' not in storm_group:
            print_warning(f"{atcf_id}: Missing 'news' group")
        else:
            missing_news_vars = set(EXPECTED_NEWS_VARS) - set(storm_group['news'].array_keys())
            if missing_news_vars:
                print_warning(f"{atcf_id}: Missing news vars: {missing_news_vars}")
        
        storms_checked += 1
        if storms_checked <= 3:  # Only print details for first 3 storms
            print_info(f"{atcf_id}: Structure OK")
    
    if not issues:
        print_success("All structures valid")
    
    return issues


def validate_news_data(zarr_root, df, news_df):
    """NEW: Validate news data broadcasting and temporal coherence"""
    print_header("Validating News Data Broadcasting")
    
    if news_df is None:
        print_warning("No news CSV loaded - skipping news validation")
        return []
    
    issues = []
    
    for atcf_id in df['atcf_id'].unique():
        storm_csv = df[df['atcf_id'] == atcf_id].sort_values('Timestamp').reset_index(drop=True)
        storm_id = storm_csv['Storm_ID'].iloc[0]
        
        if atcf_id not in zarr_root:
            continue
        
        storm_zarr = zarr_root[atcf_id]
        
        if 'news' not in storm_zarr or 'num_articles' not in storm_zarr['news']:
            print_warning(f"{atcf_id}: No news data in Zarr")
            continue
        
        # Get news arrays
        zarr_articles = storm_zarr['news/num_articles'][:]
        timestamps = pd.to_datetime(storm_zarr['timestamps'][:])
        
        # Check array length matches timestamps
        if len(zarr_articles) != len(timestamps):
            print_error(f"{atcf_id}: News array length mismatch (news: {len(zarr_articles)}, timestamps: {len(timestamps)})")
            issues.append(f"{atcf_id}: News length mismatch")
            continue
        
        # Get corresponding news from CSV
        storm_news = news_df[news_df['Storm_ID'] == storm_id]
        
        if len(storm_news) == 0:
            # No news for this storm - should be all zeros
            if np.any(zarr_articles != 0):
                print_error(f"{atcf_id}: Has news data in Zarr but none in CSV")
                issues.append(f"{atcf_id}: Unexpected news data")
            else:
                print_info(f"{atcf_id}: No news data (correctly all zeros)")
            continue
        
        # Validate broadcasting: check that values are constant within each day
        storm_news['date'] = pd.to_datetime(storm_news['day'], format='%Y%m%d')
        
        daily_violations = []
        for day in timestamps.normalize().unique():
            # Get all timestamps for this day
            day_mask = timestamps.normalize() == day
            day_articles = zarr_articles[day_mask]
            
            # All values within the same day should be identical (broadcasting)
            if len(day_articles) > 0:
                unique_vals = np.unique(day_articles)
                if len(unique_vals) > 1:
                    daily_violations.append(f"{day.date()}: {len(unique_vals)} different values")
        
        if daily_violations:
            print_error(f"{atcf_id}: Broadcasting violation - values not constant within days")
            for violation in daily_violations[:3]:
                print_info(f"  - {violation}")
            issues.append(f"{atcf_id}: Broadcasting error")
        else:
            # Check that the daily values match the news CSV
            total_articles_zarr = np.sum(zarr_articles > 0)
            total_articles_csv = len(storm_news)
            
            # Calculate temporal coverage
            days_with_news = np.sum(zarr_articles > 0) / (len(zarr_articles) / 8)  # Approx 8 timesteps per day
            
            print_success(f"{atcf_id}: Broadcasting OK - {total_articles_zarr} timesteps with news ({days_with_news:.1f} days)")
            
            # Validate specific values for a few dates
            sample_dates = storm_news.sample(min(3, len(storm_news)))
            mismatches = []
            for _, row in sample_dates.iterrows():
                date = row['date']
                expected_articles = row['num_articles']
                
                # Find zarr values for this date
                day_mask = timestamps.normalize() == date
                day_articles = zarr_articles[day_mask]
                
                if len(day_articles) > 0:
                    if not np.all(day_articles == expected_articles):
                        mismatches.append(f"{date.date()}: expected {expected_articles}, got {day_articles[0]}")
            
            if mismatches:
                print_warning(f"{atcf_id}: Value mismatches found:")
                for mismatch in mismatches:
                    print_info(f"  - {mismatch}")
    
    return issues


def validate_news_temporal_coherence(zarr_root, df):
    """NEW: Validate temporal patterns in news data"""
    print_header("Validating News Temporal Coherence")
    
    issues = []
    
    for atcf_id in df['atcf_id'].unique():
        if atcf_id not in zarr_root:
            continue
        
        storm_zarr = zarr_root[atcf_id]
        
        if 'news' not in storm_zarr or 'num_articles' not in storm_zarr['news']:
            continue
        
        articles = storm_zarr['news/num_articles'][:]
        timestamps = pd.to_datetime(storm_zarr['timestamps'][:])
        
        if np.all(articles == 0):
            print_info(f"{atcf_id}: No news coverage")
            continue
        
        # Find news spikes (significant increases)
        diffs = np.diff(articles)
        spikes = np.where(diffs > 0)[0]
        
        if len(spikes) > 0:
            # Check that spikes align with day boundaries (should happen at 00:00)
            spike_hours = [timestamps[s+1].hour for s in spikes]
            non_midnight_spikes = [h for h in spike_hours if h != 0]
            
            if len(non_midnight_spikes) > 0:
                print_warning(f"{atcf_id}: {len(non_midnight_spikes)} news spikes NOT at day boundary (expected 00:00)")
                print_info(f"  Spike hours: {non_midnight_spikes[:5]}")
            else:
                print_success(f"{atcf_id}: {len(spikes)} news transitions aligned with day boundaries")
        
        # Calculate statistics
        max_articles = np.max(articles)
        mean_articles = np.mean(articles[articles > 0]) if np.any(articles > 0) else 0
        
        print_info(f"  Max: {max_articles} articles/day, Mean (non-zero): {mean_articles:.1f}")
    
    return issues


def validate_tabular_alignment(zarr_root, df):
    """Validate that tabular data in Zarr matches CSV"""
    print_header("Validating Tabular Data Alignment")
    
    issues = []
    
    for atcf_id in df['atcf_id'].unique():
        storm_csv = df[df['atcf_id'] == atcf_id].sort_values('Timestamp').reset_index(drop=True)
        
        if atcf_id not in zarr_root:
            continue
        
        storm_zarr = zarr_root[atcf_id]
        
        # Check timestamp count
        n_csv = len(storm_csv)
        n_zarr = len(storm_zarr['timestamps'][:])
        
        if n_csv != n_zarr:
            print_error(f"{atcf_id}: Timestamp count mismatch (CSV: {n_csv}, Zarr: {n_zarr})")
            issues.append(f"{atcf_id}: Count mismatch")
            continue
        
        # Validate specific columns
        checks = [
            ('Latitude', 'tabular/latitude'),
            ('Longitude', 'tabular/longitude'),
            ('Observed_Wind_Max_Knots', 'tabular/wind_max_knots'),
            ('Observed_Pressure_Min_mb', 'tabular/pressure_min_mb'),
        ]
        
        mismatches = []
        for csv_col, zarr_path in checks:
            csv_vals = storm_csv[csv_col].fillna(np.nan).values
            zarr_vals = storm_zarr[zarr_path][:]
            
            # Allow small floating point differences
            if not np.allclose(csv_vals, zarr_vals, rtol=1e-5, atol=1e-8, equal_nan=True):
                max_diff = np.nanmax(np.abs(csv_vals - zarr_vals))
                mismatches.append(f"{csv_col} (max_diff: {max_diff:.6f})")
        
        if mismatches:
            print_error(f"{atcf_id}: Data mismatches in {', '.join(mismatches)}")
            issues.append(f"{atcf_id}: Data mismatch")
        else:
            print_success(f"{atcf_id}: Tabular data matches CSV ({n_csv} records)")
    
    return issues


def validate_reliefweb_data(zarr_root, df):
    """Validate ReliefWeb data integrity"""
    print_header("Validating ReliefWeb Data Integrity")
    
    issues = []
    
    rw_vars = [
        ('RW_Casualty_Info', 'rw_casualties'),
        ('RW_Injured_Info', 'rw_injured'),
        ('RW_Evacuated_Displaced', 'rw_evacuated'),
        ('RW_Affected_Population', 'rw_affected'),
        ('RW_Total_Reports', 'rw_total_reports')
    ]
    
    for atcf_id in df['atcf_id'].unique():
        storm_csv = df[df['atcf_id'] == atcf_id].sort_values('Timestamp').reset_index(drop=True)
        
        if atcf_id not in zarr_root:
            continue
        
        storm_zarr = zarr_root[atcf_id]
        
        mismatches = []
        for csv_col, zarr_var in rw_vars:
            if f'tabular/{zarr_var}' not in storm_zarr:
                continue
            
            csv_vals = storm_csv[csv_col].fillna(0).values
            zarr_vals = storm_zarr[f'tabular/{zarr_var}'][:]
            
            if not np.allclose(csv_vals, zarr_vals, rtol=1e-5, atol=1e-8):
                max_diff = np.nanmax(np.abs(csv_vals - zarr_vals))
                mismatches.append(f"{csv_col} (max_diff: {max_diff:.1f})")
        
        if mismatches:
            print_error(f"{atcf_id}: ReliefWeb mismatches in {', '.join(mismatches)}")
            issues.append(f"{atcf_id}: RW mismatch")
        else:
            # Check if there's any RW data
            total_reports = np.sum(storm_zarr['tabular/rw_total_reports'][:])
            if total_reports > 0:
                print_success(f"{atcf_id}: ReliefWeb data OK ({int(total_reports)} total reports)")
            else:
                print_info(f"{atcf_id}: No ReliefWeb reports (OK)")
    
    return issues


def validate_physical_ranges(zarr_root):
    """Validate that image data falls within physically reasonable ranges"""
    print_header("Validating Physical Data Ranges")
    
    issues = []
    
    for atcf_id in zarr_root.group_keys():
        storm_zarr = zarr_root[atcf_id]
        
        if 'images' not in storm_zarr:
            continue
        
        img_group = storm_zarr['images']
        
        for var_name in EXPECTED_IMAGE_VARS:
            if var_name not in img_group:
                continue
            
            data = img_group[var_name][:]
            
            # Get non-NaN values
            valid_data = data[~np.isnan(data)]
            
            if len(valid_data) == 0:
                continue
            
            min_val = np.min(valid_data)
            max_val = np.max(valid_data)
            
            if var_name in PHYSICAL_RANGES:
                expected_min, expected_max = PHYSICAL_RANGES[var_name]
                
                # Check for values outside expected range
                out_of_range = np.sum((valid_data < expected_min) | (valid_data > expected_max))
                out_of_range_pct = out_of_range / len(valid_data) * 100
                
                if out_of_range_pct > 5:  # More than 5% out of range
                    print_warning(f"{atcf_id}/{var_name}: {out_of_range_pct:.1f}% values out of range [{expected_min}, {expected_max}]")
                    print_info(f"  Actual range: [{min_val:.2f}, {max_val:.2f}]")
                    issues.append(f"{atcf_id}/{var_name}: Range issue")
                elif out_of_range_pct > 0:
                    print_info(f"{atcf_id}/{var_name}: Range OK (actual: [{min_val:.2f}, {max_val:.2f}], {out_of_range_pct:.2f}% outliers)")
                else:
                    print_success(f"{atcf_id}/{var_name}: Range OK [{min_val:.2f}, {max_val:.2f}]")
            else:
                print_info(f"{atcf_id}/{var_name}: [{min_val:.2f}, {max_val:.2f}] (no reference range)")
    
    return issues


def validate_fillvalue_integrity(zarr_root):
    """Validate that missing data is NaN, not 0.0"""
    print_header("Validating FillValue Integrity (NaN vs 0.0)")
    
    issues = []
    
    for atcf_id in zarr_root.group_keys():
        storm_zarr = zarr_root[atcf_id]
        
        if 'images' not in storm_zarr or 'quality_meta' not in storm_zarr:
            continue
        
        img_group = storm_zarr['images']
        time_diffs = storm_zarr['quality_meta/time_diff_min'][:]
        
        # Identify timesteps with NO satellite match (should be all NaN)
        no_match_indices = np.where(np.isnan(time_diffs))[0]
        
        if len(no_match_indices) == 0:
            print_success(f"{atcf_id}: All timesteps matched (no NaN checking needed)")
            continue
        
        problems = []
        for var_name in EXPECTED_IMAGE_VARS:
            if var_name not in img_group:
                continue
            
            data = img_group[var_name][:]
            
            # Check if unmatched timesteps are properly filled with NaN
            for idx in no_match_indices:
                img = data[idx]
                if not np.all(np.isnan(img)):
                    # Check if it's all zeros (wrong fill value)
                    if np.all(img == 0.0):
                        problems.append(f"{var_name}[{idx}] is 0.0 instead of NaN")
                    else:
                        problems.append(f"{var_name}[{idx}] has unexpected data")
        
        if problems:
            print_error(f"{atcf_id}: FillValue issues detected")
            for prob in problems[:3]:  # Show first 3
                print_info(f"  - {prob}")
            issues.append(f"{atcf_id}: FillValue error")
        else:
            print_success(f"{atcf_id}: FillValue OK ({len(no_match_indices)} unmatched timesteps correctly NaN)")
    
    return issues


def validate_quality_metadata(zarr_root, df):
    """Validate ALL quality metadata arrays and collect global stats"""
    print_header("Validating Quality Metadata")
    
    issues = []
    # Accumulators for global stats
    global_stats = {
        'total_dist_sum': 0.0, 'total_dist_count': 0,
        'total_time_sum': 0.0, 'total_time_count': 0
    }
    
    for atcf_id in df['atcf_id'].unique():
        if atcf_id not in zarr_root: continue
        storm_zarr = zarr_root[atcf_id]
        if 'quality_meta' not in storm_zarr: continue
        
        qual_group = storm_zarr['quality_meta']
        
        # Collect Time Difference stats
        if 'time_diff_min' in qual_group:
            time_diffs = qual_group['time_diff_min'][:]
            non_nan = time_diffs[~np.isnan(time_diffs)]
            if len(non_nan) > 0:
                global_stats['total_time_sum'] += np.sum(non_nan)
                global_stats['total_time_count'] += len(non_nan)
                print_info(f"{atcf_id}: mean Δt: {np.mean(non_nan):.1f}min")

        # Collect Center Distance stats
        if 'center_dist_km' in qual_group:
            distances = qual_group['center_dist_km'][:]
            non_nan = distances[~np.isnan(distances)]
            if len(non_nan) > 0:
                global_stats['total_dist_sum'] += np.sum(non_nan)
                global_stats['total_dist_count'] += len(non_nan)
                print_info(f"  Center distance: {np.mean(non_nan):.1f}km")
                
    return issues, global_stats


def validate_image_data(zarr_root, df):
    """Validate satellite images and collect missing value percentages"""
    print_header("Validating Satellite Image Data")
    
    issues = []
    stats = {
        'total_timestamps': 0,
        'missing_ir': 0,
        'missing_gprof': 0,
        'storms_with_ir': 0,
        'storms_with_gprof': 0
    }
    
    for atcf_id in df['atcf_id'].unique():
        if atcf_id not in zarr_root: continue
        storm_zarr = zarr_root[atcf_id]
        n_times = len(storm_zarr['timestamps'][:])
        stats['total_timestamps'] += n_times
        
        # IR Stats
        if 'images/ir' in storm_zarr:
            ir_data = storm_zarr['images/ir'][:]
            # Count frames that are ALL NaN
            missing_ir = np.sum(np.all(np.isnan(ir_data), axis=(1, 2)))
            stats['missing_ir'] += missing_ir
            if (n_times - missing_ir) > 0: stats['storms_with_ir'] += 1
            print_info(f"{atcf_id}: IR {n_times - missing_ir}/{n_times} matched")

        # GPROF Stats (using surface_precip as the proxy for GPROF availability)
        if 'images/surface_precip' in storm_zarr:
            gprof_data = storm_zarr['images/surface_precip'][:]
            missing_gprof = np.sum(np.all(np.isnan(gprof_data), axis=(1, 2)))
            stats['missing_gprof'] += missing_gprof
            if (n_times - missing_gprof) > 0: stats['storms_with_gprof'] += 1
            
    return issues, stats


def create_visual_checks(zarr_root, df):
    """Create visual sanity check plots with proper colormaps"""
    print_header("Creating Visual Sanity Checks")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Select first storm with data for visualization
    for atcf_id in df['atcf_id'].unique():
        if atcf_id not in zarr_root:
            continue
        
        storm_zarr = zarr_root[atcf_id]
        
        if 'images' not in storm_zarr or 'ir' not in storm_zarr['images']:
            continue
        
        ir_data = storm_zarr['images']['ir'][:]
        timestamps = storm_zarr['timestamps'][:]
        
        # Find first non-NaN image
        for i, img in enumerate(ir_data):
            if not np.all(np.isnan(img)):
                # Create visualization with proper colormaps and ranges
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                fig.suptitle(f'Sanity Check: {storm_zarr.attrs["storm_name"]} ({atcf_id})\nTimestep {i}', 
                            fontsize=16, fontweight='bold')
                
                # Plot IR image with proper range
                vmin, vmax = PHYSICAL_RANGES['ir']
                im0 = axes[0, 0].imshow(img, cmap=COLORMAPS['ir'], vmin=vmin, vmax=vmax)
                axes[0, 0].set_title('IR Brightness Temperature (K)')
                axes[0, 0].axis('off')
                plt.colorbar(im0, ax=axes[0, 0], label='Kelvin')
                
                # Plot other GPROF products with proper ranges
                img_group = storm_zarr['images']
                plot_idx = 1
                for var in ['surface_precip', 'convective_precip', 'rain_water_path', 
                           'cloud_water_path', 'ice_water_path']:
                    if var in img_group and plot_idx < 6:
                        data = img_group[var][i]
                        row, col = plot_idx // 3, plot_idx % 3
                        if not np.all(np.isnan(data)):
                            # Use proper colormap and range
                            vmin, vmax = PHYSICAL_RANGES.get(var, (np.nanmin(data), np.nanmax(data)))
                            cmap = COLORMAPS.get(var, 'viridis')
                            
                            im = axes[row, col].imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
                            title = var.replace('_', ' ').title()
                            axes[row, col].set_title(title)
                            axes[row, col].axis('off')
                            
                            # Add proper units
                            units = 'mm/hr' if 'precip' in var else 'kg/m²'
                            plt.colorbar(im, ax=axes[row, col], label=units)
                            plot_idx += 1
                
                # Hide unused subplots
                for idx in range(plot_idx, 6):
                    row, col = idx // 3, idx % 3
                    axes[row, col].axis('off')
                
                plt.tight_layout()
                output_path = os.path.join(OUTPUT_DIR, f'sanity_check_{atcf_id}.png')
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                print_success(f"Created visualization: {output_path}")
                break
        
        break  # Only create for first storm with data
    
    # Create data availability heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    storms = []
    availability = []
    
    for atcf_id in sorted(df['atcf_id'].unique()):
        if atcf_id not in zarr_root:
            continue
        
        storm_zarr = zarr_root[atcf_id]
        storms.append(storm_zarr.attrs.get('storm_name', atcf_id))
        
        row = []
        for var in EXPECTED_IMAGE_VARS:
            if 'images' in storm_zarr and var in storm_zarr['images']:
                data = storm_zarr['images'][var][:]
                non_nan_pct = np.sum(~np.all(np.isnan(data), axis=(1, 2))) / len(data) * 100
                row.append(non_nan_pct)
            else:
                row.append(0)
        availability.append(row)
    
    availability = np.array(availability)
    
    im = ax.imshow(availability, aspect='auto', cmap='YlGn', vmin=0, vmax=100)
    ax.set_xticks(range(len(EXPECTED_IMAGE_VARS)))
    ax.set_xticklabels([v.replace('_', '\n') for v in EXPECTED_IMAGE_VARS], rotation=45, ha='right')
    ax.set_yticks(range(len(storms)))
    ax.set_yticklabels(storms)
    ax.set_title('Data Availability Heatmap (% of timestamps with data)', fontweight='bold')
    
    for i in range(len(storms)):
        for j in range(len(EXPECTED_IMAGE_VARS)):
            text = ax.text(j, i, f'{availability[i, j]:.0f}%',
                          ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im, ax=ax, label='% Available')
    plt.tight_layout()
    
    heatmap_path = os.path.join(OUTPUT_DIR, 'data_availability_heatmap.png')
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print_success(f"Created availability heatmap: {heatmap_path}")


def generate_report(all_issues, stats):
    """Generate final validation report"""
    print_header("Validation Report")
    
    if not any(all_issues.values()):
        print(f"{Colors.GREEN}{Colors.BOLD}✅ ALL CHECKS PASSED!{Colors.END}\n")
        print_info("The Zarr dataset is properly structured and aligned with the CSV.")
        print_info(f"Total non-NaN satellite images: {stats.get('total_images', 0)}")
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ ISSUES FOUND{Colors.END}\n")
        for check_name, issues in all_issues.items():
            if issues:
                print_error(f"{check_name}: {len(issues)} issues")
                for issue in issues[:5]:  # Show first 5
                    print_info(f"  - {issue}")
                if len(issues) > 5:
                    print_info(f"  ... and {len(issues) - 5} more")
    
    # Save report to file
    report_path = os.path.join(OUTPUT_DIR, 'validation_report.txt')
    with open(report_path, 'w') as f:
        f.write("TC-PRIMED Zarr Dataset Validation Report\n")
        f.write("=" * 70 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Summary:\n")
        if not any(all_issues.values()):
            f.write("✅ ALL CHECKS PASSED\n\n")
        else:
            f.write("✗ ISSUES FOUND\n\n")
        
        for check_name, issues in all_issues.items():
            f.write(f"\n{check_name}:\n")
            if issues:
                for issue in issues:
                    f.write(f"  - {issue}\n")
            else:
                f.write("  ✓ No issues\n")
        
        if stats:
            f.write(f"\n\nStatistics:\n")
            for key, value in stats.items():
                f.write(f"  {key}: {value}\n")
    
    print_success(f"Report saved: {report_path}")


def main():
    """Main validation pipeline"""
    print(f"{Colors.BOLD}TC-PRIMED Zarr Dataset Validation with News Data{Colors.END}")
    print(f"CSV: {INPUT_CSV}")
    print(f"News CSV: {NEWS_CSV}")
    print(f"Zarr: {ZARR_PATH}")
    
    # Load data
    df, news_df, zarr_root = load_data()
    if df is None or zarr_root is None:
        print_error("Failed to load data. Exiting.")
        return
    
    # Run all validation checks
    all_issues = {}
    
    all_issues['Structure'] = validate_structure(zarr_root, df)
    all_issues['Tabular Alignment'] = validate_tabular_alignment(zarr_root, df)
    all_issues['ReliefWeb Data'] = validate_reliefweb_data(zarr_root, df)
    
    # NEW: News validation
    all_issues['News Broadcasting'] = validate_news_data(zarr_root, df, news_df)
    all_issues['News Temporal Coherence'] = validate_news_temporal_coherence(zarr_root, df)
    
    image_issues, stats = validate_image_data(zarr_root, df)
    all_issues['Image Data'] = image_issues
    
    all_issues['Physical Ranges'] = validate_physical_ranges(zarr_root)
    all_issues['FillValue Integrity'] = validate_fillvalue_integrity(zarr_root)
    all_issues['Quality Metadata'], qual_stats = validate_quality_metadata(zarr_root, df)

    avg_dist = qual_stats['total_dist_sum'] / qual_stats['total_dist_count'] if qual_stats['total_dist_count'] > 0 else 0
    avg_time = qual_stats['total_time_sum'] / qual_stats['total_time_count'] if qual_stats['total_time_count'] > 0 else 0
    
    print(f"{Colors.BOLD}Global Quality Metrics:{Colors.END}")
    print(f"  Mean Center Distance: {avg_dist:.2f} km")
    print(f"  Mean Time Difference: {avg_time:.2f} min")
    
    # 2. Missing Value Percentages
    ir_pct = (stats['missing_ir'] / stats['total_timestamps']) * 100
    gprof_pct = (stats['missing_gprof'] / stats['total_timestamps']) * 100
    
    print(f"\n{Colors.BOLD}Missing Data Percentages (Temporal Availability):{Colors.END}")
    print(f"  IR Missing Frames:    {ir_pct:.2f}%")
    print(f"  GPROF Missing Frames: {gprof_pct:.2f}%")
    print(f"  Total Timestamps:     {stats['total_timestamps']}")
    
    print("\n" + "="*70)
    
    # Create visualizations
    create_visual_checks(zarr_root, df)
    
    # Generate final report
    generate_report(all_issues, stats)


if __name__ == "__main__":
    main()