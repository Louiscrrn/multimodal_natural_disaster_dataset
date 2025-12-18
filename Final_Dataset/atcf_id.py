import pandas as pd

df = pd.read_csv('../data/processed/ibtracs_era5_20251217_1724.csv', keep_default_na=False)

# Map ocean basin to ATCF basin
basin_map = {
    'NA': 'NA',
    'EP': 'EP',
    'WP': 'WP',
    'NI': 'NI',
    'SP': 'SH',
    'SI': 'SH',
    'SA': 'SH'
}
df['ATCF_basin'] = df['Ocean_Basin'].map(basin_map)

# Step 1: one row per storm
unique_storms = df[['Storm_ID', 'Year', 'ATCF_basin', 'number', 'Timestamp']].copy()
unique_storms = unique_storms.sort_values('Timestamp')
unique_storms = unique_storms.drop_duplicates(subset='Storm_ID', keep='first')

# Step 2: sort by year and basin, then assign basin-relative number
unique_storms = unique_storms.sort_values(['Year', 'ATCF_basin', 'number'])
unique_storms['ATCF_number'] = unique_storms.groupby(['Year', 'ATCF_basin']).cumcount() + 1

# Step 3: build ATCF_id
unique_storms['ATCF_id'] = unique_storms['ATCF_basin'] + unique_storms['ATCF_number'].astype(str).str.zfill(2) + unique_storms['Year'].astype(int).astype(str)

# Step 4: merge back to original dataset
df = df.merge(unique_storms[['Storm_ID', 'ATCF_id']], on='Storm_ID', how='left')

df.to_csv('../data/processed/ibtracs_era5_20251217_2118a.csv', index=False)