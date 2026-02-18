# IDOBE: Infectious Disease Outbreak forecasting Benchmark Ecosystem

## Overview 
IDOBE is a curated collection of epidemiological time series focused on outbreak forecasting. IDOBE compiles from multiple data repositories spanning over a century of surveillance and across U.S. states and global locations. This repository provides 
- Over 10000 outbreak timeseries corresponding to different diseases
- Scripts for extracting analytical measures to analyze outbreaks
- A suite of trained baseline forecasting models (statistical and DNN models)
- Evaluation scripts of probabilistic forecasts

## Data
- `raw_data/outbreaks_disease_location.csv`
## Data statistics
- `stats/output/` - consists of entropy, permutation entropy, and shape statistics for the different outbreaks
- `stats/stats_compute.ipynb` - notebook for computing the different statistics
## Baseline models
- `baselines/src/` - folder contains different classes of models
