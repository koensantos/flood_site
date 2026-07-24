# Hoboken Flood Risk Portfolio Dashboard

A production-style, portfolio-ready flood risk analysis project for Hoboken, New Jersey. The repository combines GIS, hydrologic, and machine-learning workflows to estimate flood susceptibility from the same processed dataset used in the project, with rainfall amount as the interactive input.

## Project Summary

This project demonstrates a practical flood-risk modeling workflow using:

- LiDAR / DEM elevation data
- FEMA flood zone geometry
- USGS high-water mark observations
- Newark airport rainfall observations
- Random Forest and XGBoost classification models

The key interactive deliverable is a Streamlit dashboard that lets the user adjust rainfall input and immediately view the modeled flood amount / risk surface for the Hoboken study area.

## What the App Shows

The dashboard presents:

- a live flood amount image
- average modeled flood probability
- high-risk cell count
- risk distribution by severity
- a scatter view of risk versus elevation and distance from FEMA flood zones
- the top highest-risk cells in the current scenario

## Data Sources

The app uses the processed project outputs already generated in the repository:

- `output_csvs/feature_grid.csv`
- `rainfall_dataset/lcd_newark_sept2021.csv`
- `fema_flood_zones_dataset/S_FLD_HAZ_AR.shp`

## Repository Structure

- `flood_portfolio_app.py` — interactive Streamlit portfolio dashboard
- `visualize_results.py` — renders static flood and comparison maps
- `process_features.py` — creates the feature grid used in the modeling workflow
- `train_model.py` — trains the Random Forest and XGBoost models
- `output_csvs/` — saved predictions and metrics
- `map_flood_amount.png` — primary flood intensity image produced by the projection workflow

## Portfolio Value

This project is suitable for:

- geospatial ML portfolio demos
- disaster risk dashboards
- environmental data science presentations
- data-driven storytelling around flood exposure and urban resilience

## Environment Setup

Use Python 3.10+ and install the project dependencies.

### Option 1: pip

```powershell
pip install -r requirements.txt
```

### Option 2: manual install

```powershell
pip install streamlit pandas numpy matplotlib scikit-learn xgboost geopandas rasterio shapely
```

## Run the Portfolio App

From the project root, run:

```powershell
python -m streamlit run flood_portfolio_app.py --server.headless true --server.port 8502
```

Then open:

```text
http://localhost:8502
```

## Regenerate the Static Maps

To recreate the PNG outputs such as the flood heat map and comparison figures:

```powershell
python visualize_results.py
```

## Model Notes

The dashboard uses the same Hoboken flood-risk dataset already processed in this repository. The `rainfall_inches` input is varied inside the app to generate a new scenario view of the modeled flood surface.

## Production Readiness Notes

This repository is organized to support:

- local interactive presentation
- reproducible model and feature generation
- reusable output artifacts for portfolio display
- a straightforward handoff for deployment or further enhancement

## Output Artifacts

The main generated artifacts include:

- `map_flood_amount.png`
- `map_flood_probability.png`
- `map_predictions_vs_fema.png`
- `output_csvs/model_predictions.csv`
- `output_csvs/model_metrics.csv`

## Questions / Next Improvements

Possible future enhancements include:

- adding a real map basemap instead of grid scatter rendering
- supporting Jersey City as a second selectable geography
- packaging the app for cloud deployment
- exposing model confidence intervals and threshold tuning controls

## License

This project is intended for educational and portfolio demonstration use.
