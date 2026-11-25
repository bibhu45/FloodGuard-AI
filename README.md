# FloodGuard AI — Interactive Streamlit Dashboard

This project is built around the actual files in `data/FloodGuard_Data/` from your uploaded dataset ZIP.

## Important data note
All datasets are synthetic. This first dashboard uses **synthetic event replay**, not ConvLSTM-generated forecast outputs. It does not invent model metrics. The Model Performance page stays unpopulated until evaluation results are computed.

The flood-risk visualization is a prototype relative index based on rainfall, elevation, land cover, and river proximity. It is **not flood depth** and is not a validated hydraulic model.

## Run locally (Windows)
1. Extract this project ZIP.
2. Open a terminal in the `FloodGuard_AI_Project` folder.
3. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Start the dashboard:
   ```bash
   streamlit run app.py
   ```

## Five navigation sections
- Dashboard
- Rainfall Forecast
- Flood Risk Map
- Model Performance
- About FloodGuard AI

## Next integration step
After training ConvLSTM and generating test metrics, save forecast arrays and metric files under `outputs/forecasts/` and `outputs/metrics/`, then update the forecast and evaluation loaders to use those outputs.


## Optional trained output integration

### Forecast output
For each replay event, save:
`outputs/forecasts/forecast_YYYY-MM-DD.npz`

The NPZ should contain:
- `rain_forecast`: array with shape `(6, 20, 20)` in mm/hr

The Forecast page will display this as a saved ConvLSTM forecast. If the file is absent, the page uses labelled synthetic event replay.

### Metrics output
Save computed results to:
`outputs/metrics/metrics.csv`

Required columns:
- `model` (ConvLSTM, Persistence, Optical Flow)
- `lead_minutes` (30, 60, 90, 120, 150, 180)
- `POD`, `FAR`, `CSI`, `FSS`

Optional column:
- `threshold_mmhr`

The performance page will plot these values. It does not fabricate missing metrics.
