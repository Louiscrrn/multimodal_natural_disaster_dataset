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