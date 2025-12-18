# Multimodal Natural Disaster DataSet

This project aims to build a clean, reproducible, and multimodal dataset for natural disasters by integrating heterogeneous sources such as cyclone tracks, meteorological reanalysis, satellite data, and textual reports.

## Project Structure

```
multimodal_natural_disaster_dataset/
├── data/
│   ├── raw/
│   │   ├── IBTrACS.ALL.v04r01.nc
│   │   └── era5_yearly/
│   │       ├── era5_2021_SI.nc
│   │       ├── era5_2022_*.nc
│   │       └── ...
│   └── processed/
│       └── ibtracs_era5_*.csv
├── ERATrACS/
│   ├── download_era5.py
│   ├── sample_era5_existing.py
│   └── notebooks/
│       ├── analysis.ipynb
│       └── ...
├── GDELT/
│   ├── gdelt_loading_files.py 
│   ├── gdelt_cyclone_utils.py   
│   ├── gdelt_cyclone_pipeline.py
│   ├── get_daily_mentions.py
├── README.md
└── requirements.txt
```


## ERATrACS

ERATrACS is a data preprocessing module that integrates IBTrACS cyclone track data with ERA5 reanalysis meteorological variables. It produces a clean, tabular dataset in CSV format, suitable for analysis, modeling, and research on tropical cyclones.

### Features

- **Data Integration**: Merges cyclone tracks from IBTrACS with ERA5 reanalysis data.
- **Temporal Resolution**: Maintains IBTrACS observation timestamps.
- **Spatial Sampling**: Uses nearest-neighbor interpolation on the ERA5 grid.
- **Quality Assurance**: Includes checks for missing values, coverage ratios, duplicates, and spatial errors.
- **Reproducibility**: Supports configurable parameters for years, basins, and regions.


### Downloading ERA5 Data

Use `download_era5.py` to fetch ERA5 data for specified years and basins. The script restricts downloads to cyclone-active periods and predefined basin boundaries.

```bash
python ERATrACS/download_era5.py --years 2022 2023 2024 --basins NA EP WP
```

Output files are saved to `data/raw/era5_yearly/era5_<year>_<basin>.nc`.

### Processing and Sampling

Run `sample_era5_existing.py` to process IBTrACS data and sample ERA5 variables at cyclone locations.

```bash
python ERATrACS/sample_era5_existing.py --years 2022 2023 2024 --basins NA EP WP
```

Output files are saved to `data/processed/ibtracs_era5_<timestamp>.csv`.

### Data Sources

- **IBTrACS**: International Best Track Archive for Climate Stewardship (NOAA). Provides cyclone tracks, metadata, and observations.
- **ERA5**: European Centre for Medium-Range Weather Forecasts (ECMWF) reanalysis data. Includes variables such as 2m temperature, mean sea level pressure, and 10m wind components (u, v).

### Quality Checks

The pipeline performs several validation steps:

- Missing value analysis per year and variable.
- Coverage ratios comparing IBTrACS observations to ERA5 availability.
- Detection of duplicate or conflicting records (based on Storm_ID and Timestamp).
- Spatial error assessment using Haversine distance, validated against ERA5 grid resolution.

### Notebooks

Exploratory and analysis notebooks are available in `ERATrACS/notebooks/` and `ERATrACS/other/`:
- `analysis.ipynb`: Main analysis notebook.

## GDELT

The GDELT module complements the physical cyclone dataset by extracting news media coverage related to tropical cyclones from the Global Database of Events, Language, and Tone (GDELT).
This enables a multimodal perspective, linking observed cyclone evolution with how events are reported in the media over time.

This module identifies news articles associated with cyclones using:

* Cyclone names (when available),
* Storm-related keywords (e.g., cyclone, hurricane, typhoon),
* Basin-specific geographic keywords,
* Geographic validation using GDELT Global Knowledge Graph (GKG).

The resulting dataset provides daily aggregated media mentions for each cyclone, including article URLs and sources.

### Data Source
GDELT: Global Database of Events, Language, and Tone

* High-frequency global news monitoring
* Updates every 15 minutes
* Includes article metadata, URLs, sources, and extracted locations

### Methodology

The extraction pipeline follows a two-stage filtering strategy:

1- Fast keyword filtering (Mentions table)

* Searches for cyclone names and storm-related keywords in article titles and sources
* Uses word boundaries to avoid false positives (e.g., brainstorm)
* Applies basin-specific geographic keywords to limit unrelated events

2- Geographic validation (GKG)

* When the ocean basin is unknown, candidate articles are validated using extracted locations
* Articles are retained only if mentioned coordinates fall within a bounding box around the cyclone track

To balance completeness and computational cost, GDELT files are processed at 3-hour intervals, and results are aggregated at the daily scale, with duplicate articles removed based on URL endings.

### Running the pipeline
To generate the final cyclone–media dataset, run:
```bash
python GDELT/get_daily_mentions.py
```

This script:

* Loads processed IBTrACS–ERA5 cyclone data
* Builds a ±3-day temporal window around each cyclone timestamp
* Downloads and filters relevant GDELT files
* Aggregates unique articles per cyclone and per day

### Output

The final dataset is saved to:
```bash
data/processed/cyclones_mentions_gdelt_3h_2022_2023.csv
```
Each row corresponds to a cyclone–day pair and includes:
* Cyclone name
* Date
* List of article URLs
* List of news sources
* Number of unique articles

This table can be directly merged with the ERATrACS output to support joint physical–media analyses of tropical cyclones.

## ReliefWeb

The ReliefWeb module adds a humanitarian layer to the dataset by extracting "ground truth" impact data (fatalities, injuries, displacement) from Situation Reports published by the UN Office for the Coordination of Humanitarian Affairs (OCHA). This allows for the correlation of physical cyclone intensity with real-world human consequences.

### Features

- **Relevance Filtering**: Uses a scoring algorithm to distinguish between specific storm reports and generic humanitarian documents.
- **Multidimensional Impact**: Extracts data on casualties, injuries, displacement, and disease outbreaks.
- **Source Traceability**: Every extracted data point is linked to its original report URL for verification.
- **Broadcasting**: Propagates the global impact of a storm to all its track points (temporal broadcasting) to facilitate track-based analysis.

### Methodology

The enrichment pipeline processes the physical track data through four stages:

1.  **Aggregation**: Identifies unique storms (Name + Year) from the processed IBTrACS dataset.
2.  **API Querying**: Iterates through multiple search strategies (e.g., *"Cyclone [Name] [Year]"*, *"Hurricane [Name]"*) to maximize report retrieval from the ReliefWeb API.
3.  **Intelligent Scoring**: Filters reports based on title matches, keyword density, and temporal alignment. Reports with low relevance scores (e.g., "Annual Funding Appeal") are discarded.
4.  **Regex Extraction**: Applies Natural Language Processing (Regex) to extract numerical data associated with specific contexts (e.g., "killed", "evacuated", "cholera").

### Output

The enriched dataset is saved to:
```bash
data/processed/cyclones_2022_2023_enriched.csv
```
It appends the following humanitarian metrics to the physical track data:

* RW_Casualty_Info: Reported death toll or context phrase.
* RW_Injured_Info: Count of injured persons.
* RW_Evacuated_Displaced: Displacement and evacuation figures.
* RW_Affected_Population: Estimates of the total affected population.
* RW_Disease_Context: Mentions of post-disaster health issues (e.g., cholera, dengue).
* RW_Report_Link: Direct URL to the source report.

Usage
To run the enrichment pipeline:

```Bash

python ReliefWeb/enrich_cyclone_data.py
```

## TC-PRIMED & Zarr Integration

The final stage of the pipeline merges the tabular physical data (ERATrACS) and the news media data (GDELT) with high-resolution satellite imagery from the **NOAA TC-PRIMED** dataset. The result is a unified, multimodal dataset stored in **Zarr** format, optimized for deep learning and large-scale spatio-temporal analysis.

### Features

* **Multimodal Alignment**: Synchronizes ERA5 atmospheric data, GDELT news metrics, and TC-PRIMED satellite imagery into a single object.
* **Satellite Data Extraction**: Automatically fetches and processes passive microwave (GPROF) and infrared (IR) data from NOAA’s S3 bucket.
* **Spatial Standardization**: Resizes heterogeneous satellite swaths to a uniform 224x224 grid using bilinear interpolation.
* **Temporal Matching**: Implements a 1.5-hour time-tolerance window to pair cyclone track observations with the nearest available satellite overpass.
* **Quality Metadata**: Tracks spatial and temporal offsets, such as the distance between the storm center and the satellite footprint, to ensure data integrity.

### Data Processing Pipeline

The script `tc4.py` performs the following operations:

1.  **Tabular Consolidation**: Loads the ERATrACS CSV and aligns it with daily GDELT news mentions.
2.  **S3 Querying**: Connects to the `noaa-nesdis-tcprimed-pds` S3 bucket to locate NetCDF files corresponding to specific ATCF storm IDs and years.
3.  **Image Processing**:
    * **Masking**: Filters sentinel and fill values (e.g., -9999.0) from IR and GPROF products.
    * **Subgroup Selection**: Prioritizes GPROF subgroups (S1, S2, S3) to ensure consistent variable availability.
    * **Resizing**: Normalizes all image products (Precipitation, Water Path, IR) to a fixed 224x224 shape.
4.  **Zarr Serialization**: Writes data into a hierarchical Zarr structure, using chunks optimized for per-timestamp access.

### Dataset Structure (Zarr)

The output `TC_Clean_Dataset.zarr` is organized by `atcf_id` (e.g., `AL092022`):

| Group | Description |
| :--- | :--- |
| **`images/`** | 3D arrays `(time, 224, 224)` for `ir`, `surface_precip`, `convective_precip`, `ice_water_path`, etc. |
| **`tabular/`** | 1D arrays for physical metrics (wind speed, pressure, ERA5 variables) and ReliefWeb impact data. |
| **`news/`** | Daily GDELT article counts, encoded URL lists, and news source metadata. |
| **`quality_meta/`** | Metadata including `time_diff_min` and `center_dist_km` for satellite alignment. |

### Usage

To build the final multimodal dataset, ensure you have S3 access configured and run:

```bash
python tc4.py

## Contributing

This project is part of an academic coursework (DAES, 4A). For contributions or modifications, ensure adherence to the reproducibility requirements and data quality standards.

## Prerequisites

- Python ≥ 3.9
- Copernicus Climate Data Store (CDS) API key (configure `~/.cdsapirc`)

## Dependencies

Install the required packages using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```
