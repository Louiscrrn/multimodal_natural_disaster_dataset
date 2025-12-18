import pandas as pd
from gdelt_cyclone_utils import extract_cyclone_mentions, process_cyclone, basin_keywords

def run_pipeline(df_cyclones, gdelt_mentions_files, gdelt_gkg_files, delta_deg=5, output_file="cyclones_mentions_gdelt_3h.csv"):
    seen_url_ends = set()
    all_results = []

    active_dates = sorted(df_cyclones['date'].unique())

    for date_str in active_dates:
        try:
            print(f"Processing {date_str}")
            files = sorted([f for f in gdelt_mentions_files if f.startswith(date_str)])
            
            cyclone_row = df_cyclones[df_cyclones['date'] == date_str].iloc[0]
            cyclone_name = cyclone_row['Storm_Name'].lower() if cyclone_row['Storm_Name'].lower() != "unnamed" else None
            lat_c, lon_c = cyclone_row['Latitude'], cyclone_row['Longitude']
            
            basin = cyclone_row.get('Ocean_Basin', None)
            region_keywords = basin_keywords.get(basin, None)
            
            for f in files:
                url = f"http://data.gdeltproject.org/gdeltv2/{f}"
                df_file = extract_cyclone_mentions(url, cyclone_name, region_keywords)
                if df_file.empty:
                    continue
                
                df_file = df_file[~df_file['url_end'].isin(seen_url_ends)]
                if df_file.empty:
                    continue
                
                seen_url_ends.update(df_file['url_end'].tolist())
                all_results.append(df_file)
            
            print(f"  Found {len(all_results[-1]) if all_results else 0} articles for {date_str}")
        
        except Exception as e:
            print(f"Error processing {date_str}: {e}")
            if all_results:
                df_partial = pd.concat(all_results, ignore_index=True)
                df_partial.to_csv("cyclones_mentions_partial.csv", index=False)
                print(f"Partial results saved with {len(df_partial)} articles")
    
    df_final = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    
    df_final['event_date'] = df_final['event_date'].astype(str)

    # Extract day from 'event_date'
    df_final['day'] = df_final['event_date'].str[:8]

    # Group by 'day' and aggregate URLs and sources into lists
    daily_agg = df_final.groupby('day').agg({
        'url': lambda x: list(x),
        'source': lambda x: list(x),
        'cyclone_name': 'first',
        'EventID': 'count'         # number of articles
    }).reset_index()

    # index by cyclone_name and day
    daily_agg.set_index(['cyclone_name', 'day'], inplace=True)
    daily_agg.rename(columns={'EventID': 'num_articles'}, inplace=True)

    daily_agg.to_csv(output_file)
    return daily_agg
