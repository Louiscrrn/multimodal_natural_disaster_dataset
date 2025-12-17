import pandas as pd
import requests, zipfile, io, re

# -------------------------------
# Basin keywords
# -------------------------------
basin_keywords = {
    'SP': ['Australia', 'Fiji', 'New Caledonia', 'Solomon Islands', 'Vanuatu', 'Papua New Guinea', 'Tahiti', 'New Zealand', 'Pacific', 'South Pacific'],
    'SI': ['Madagascar', 'Mauritius', 'Reunion', 'Mozambique', 'Comoros', 'Seychelles', 'East Africa', 'South Africa', 'Indian Ocean'],
    'NI': ['Caribbean', 'Bahamas', 'Florida', 'Mexico', 'Central America', 'Puerto Rico', 'Cuba', 'Jamaica', 'Yucatan', 'Gulf of Mexico', 'Indian Ocean'],
    'WP': ['Philippines', 'Japan', 'Taiwan', 'China', 'Vietnam', 'Micronesia', 'Hong Kong', 'South Korea', 'Guam', 'Mariana Islands', 'Southeast Asia', 'Western Pacific'],
    'EP': ['Mexico', 'California', 'Central America', 'Hawaii', 'US West Coast', 'Gulf of California', 'Eastern Pacific']
}

storm_keywords_base = [
    "cyclone", "storm", "hurricane", "typhoon", "tropical storm", "severe storm"
]

text_cols = [4,5]  # source + title/url

# -------------------------------
# Extract cyclone mentions from a GDELT mentions.CSV file
# -------------------------------
def extract_cyclone_mentions(url, cyclone_name, region_keywords=None):
    results = []
    storm_keywords = ["cyclone", "storm", "hurricane", "typhoon"]
    
    if cyclone_name:
        storm_keywords = [cyclone_name, f"cyclone {cyclone_name}", f"hurricane {cyclone_name}", f"typhoon {cyclone_name}"] + storm_keywords
    
    # Word boundaries to avoid false positives like 'brainstorm'
    storm_patterns = [re.compile(rf"\b{k}\b", re.IGNORECASE) for k in storm_keywords]
    
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        return pd.DataFrame()
    
    z = zipfile.ZipFile(io.BytesIO(r.content))
    for fname in z.namelist():
        with z.open(fname) as f:
            df = pd.read_csv(f, sep="\t", header=None, dtype=str, on_bad_lines="skip")
            
            mask = pd.Series(False, index=df.index)
            for pat in storm_patterns:
                mask |= df[text_cols].apply(lambda col: col.astype(str).str.contains(pat, na=False)).any(axis=1)
            
            if region_keywords:
                region_patterns = [re.compile(rk, re.IGNORECASE) for rk in region_keywords]
                mask_region = pd.Series(False, index=df.index)
                for pat in region_patterns:
                    mask_region |= df[text_cols].apply(lambda col: col.astype(str).str.contains(pat, na=False)).any(axis=1)
                # Always keep exact cyclone name matches
                mask = mask | df[text_cols].apply(lambda col: col.astype(str).str.contains(cyclone_name, case=False, na=False)).any(axis=1)
                mask &= mask_region
            
            if mask.any():
                sub = df.loc[mask, [0,1,2,4,5]].copy()
                sub.columns = ["EventID","mention_timestamp","event_date","source","url"]
                sub["cyclone_name"] = cyclone_name
                sub["file_timestamp"] = fname[:14]
                
                # Keep only one copy per URL end
                sub['url_end'] = sub['url'].apply(lambda u: u.rstrip('/').split('/')[-1])
                sub = sub.drop_duplicates(subset='url_end', keep='first')
                
                results.append(sub)
    
    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame()


# -------------------------------
# Filter candidate mentions using GKG
# -------------------------------
def filter_mentions_with_gkg(gkg_url, candidates_df, cyclone_bbox):
    lat_min, lat_max, lon_min, lon_max = cyclone_bbox
    try:
        r = requests.get(gkg_url, timeout=120)
        r.raise_for_status()
    except:
        return pd.DataFrame()

    z = zipfile.ZipFile(io.BytesIO(r.content))
    gkg_results = []

    for fname in z.namelist():
        with z.open(fname) as f:
            try:
                gkg = pd.read_csv(f, sep="\t", header=None, dtype=str,
                                  on_bad_lines='skip', encoding='ISO-8859-1')
            except:
                continue

            col_locations = 5
            if col_locations not in gkg.columns:
                continue

            gkg_subset = gkg[gkg[0].isin(candidates_df['EventID'])].copy()
            if gkg_subset.empty:
                continue

            def in_bbox(loc_str):
                if pd.isna(loc_str):
                    return False
                for loc in loc_str.split(";"):
                    parts = loc.split(",")
                    if len(parts) < 2:
                        continue
                    try:
                        lat = float(parts[0])
                        lon = float(parts[1])
                        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                            return True
                    except:
                        continue
                return False

            mask_bbox = gkg_subset[col_locations].apply(in_bbox)
            if mask_bbox.any():
                matched_ids = gkg_subset.loc[mask_bbox, 0].unique()
                gkg_results.append(candidates_df[candidates_df['EventID'].isin(matched_ids)])

    if gkg_results:
        return pd.concat(gkg_results, ignore_index=True)
    return pd.DataFrame()


# -------------------------------
# Process cyclone for a single date
# -------------------------------
def process_cyclone(date_str, cyclone_name, lat, lon, basin, gdelt_files_mentions, gdelt_files_gkg, delta_deg=5):
    bbox = (lat - delta_deg, lat + delta_deg, lon - delta_deg, lon + delta_deg)
    region_words = basin_keywords.get(basin, None) if pd.notna(basin) else None

    mentions_all = []
    files_mentions = [f for f in gdelt_files_mentions if f.startswith(date_str)]
    for f in files_mentions:
        url = f"http://data.gdeltproject.org/gdeltv2/{f}"
        df_fast = extract_cyclone_mentions(url, cyclone_name)
        if not df_fast.empty:
            mentions_all.append(df_fast)
    
    if not mentions_all:
        return pd.DataFrame()
    candidates_df = pd.concat(mentions_all, ignore_index=True)

    if pd.isna(basin):
        gkg_files = [f for f in gdelt_files_gkg if f.startswith(date_str)]
        gkg_filtered = []
        for gf in gkg_files:
            gkg_url = f"http://data.gdeltproject.org/gdeltv2/{gf}"
            df_gkg = filter_mentions_with_gkg(gkg_url, candidates_df, bbox)
            if not df_gkg.empty:
                gkg_filtered.append(df_gkg)
        if gkg_filtered:
            return pd.concat(gkg_filtered, ignore_index=True)

    return candidates_df
