# FloodGuard AI - complete SYNTHETIC data package

**Everything in this zip is SYNTHETIC (computer-generated). It is not real weather, terrain or flood history.**
It exists so you can build and demonstrate the whole pipeline. Do not present scores from it as real accuracy.
Say on the slides: "prototype demonstrated on simulated data; real IMERG/radar/gauge integration is the next step".
The flood labels and the NWP forecast come from the same generator as the rain, so evaluation on them is partly circular.

## Contents
| Folder | File(s) | What it is |
|---|---|---|
| 01_rainfall | floodguard_synthetic_rain.npz | Main satellite-like rainfall: (23424, 20, 20) = (time, lat, lon), mm/hr, every 30 min, 1 Jun-30 Sep of 2021-2024. split: train 2021-22, val 2023, test 2024 |
| 01_rainfall | synthetic_daily_summary.csv, preview.png | Daily summary (opens in Excel) and a picture |
| 02_terrain_and_landcover | synthetic_static_layers.npz | 100x100 elevation_m, slope_deg, landcover (0 sea, 1 built-up, 2 open, 3 forest), river_mask |
| 03_flood_spots_and_events | test_events_SYNTHETIC.csv | 4 heavy-rain days in 2024 to replay in the dashboard |
| 06_gauges_and_weather | stations_SYNTHETIC.csv, gauge_weather_hourly_SYNTHETIC.csv | 15 stations, hourly rain, temperature, humidity, pressure, wind (2% missing, noisy) = the "observational data" |
| 07_nwp_forecast | nwp_forecast_SYNTHETIC.npz | Coarse 10x10 model forecast, initialised twice a day, leads 3-24 h in 3 h steps, biased and displaced |
| 08_city_boundary_and_wards | city_boundary / wards (GeoJSON), ward_terrain_summary csv | City outline, 24 ward polygons and their terrain stats |
| 09_flood_labels | flood_spots_SYNTHETIC.csv, flood_outcomes_daily_SYNTHETIC.csv | 12 flood-prone points and whether each flooded each day (about 7% of spot-days) |
| 10_radar_like_frames | radar_like_frames_SYNTHETIC.npz | 4 events at 10-min steps on a finer 40x40 grid, "radar-like" |
| 11_second_city | city_B files | A second made-up city (flat, sea on east) for the city selector |
| 12_rules | warning_thresholds, flood_risk_classes | Project-defined thresholds (not official IMD values) |
| 05_tools | open_dataset.py, FloodGuard_Dataset_Checklist.xlsx | Script that opens everything, and the checklist |

## How to open
- .csv: double-click (Excel). .png: double-click. .geojson: open at geojson.io or load in code.
- .npz: NumPy files. Install Python, run `pip install numpy pandas matplotlib`, open a terminal in 05_tools and run `python open_dataset.py`.
  In code: `d = np.load("floodguard_synthetic_rain.npz"); rain = d["rain_mm_per_hr"]`.

## Rules for using it
- Build 6-in / 6-out sequences inside each season only (never across 30 Sep -> 1 Jun).
- Train on split=train, tune on val, test on test. Never split at random.
- Area (same for all Mumbai-style files): lat 18.15-20.05, lon 71.95-73.85. Wards lie inside lat 18.85-19.30, lon 72.84-73.14.
- Flood layer must be called "flood risk", not flood depth.
- The synthetic terrain has a made-up coast, hills and river; it does NOT match real Mumbai.
