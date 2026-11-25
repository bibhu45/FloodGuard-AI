"""Open and look at ALL FloodGuard data.  Run from inside the 05_tools folder:
      python open_dataset.py
Needs:  pip install numpy pandas matplotlib      (all free)
Every dataset here is SYNTHETIC (computer-generated)."""
import json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

rain = np.load("../01_rainfall/floodguard_synthetic_rain.npz")
R = rain["rain_mm_per_hr"]                       # (time, lat, lon) mm/hr, 30-min steps
print("RAIN", R.shape, rain["time"][0], "->", rain["time"][-1], set(rain["split"]))

st = np.load("../02_terrain_and_landcover/synthetic_static_layers.npz")
print("STATIC", {k: st[k].shape for k in st.keys()})

g = pd.read_csv("../06_gauges_and_weather/gauge_weather_hourly_SYNTHETIC.csv")
print("GAUGES", g.shape, "stations:", g.station_id.nunique()); print(g.head(3))

nwp = np.load("../07_nwp_forecast/nwp_forecast_SYNTHETIC.npz")
print("NWP", nwp["nwp_rain_mm_per_hr"].shape, "(init, lead, lat, lon); leads (h):", nwp["lead_hours"])

wards = json.load(open("../08_city_boundary_and_wards/wards_SYNTHETIC.geojson"))
print("WARDS", len(wards["features"]), "polygons")

spots = pd.read_csv("../09_flood_labels/flood_spots_SYNTHETIC.csv")
lab = pd.read_csv("../09_flood_labels/flood_outcomes_daily_SYNTHETIC.csv")
print("FLOOD LABELS", lab.shape, "flooded share:", round(lab.flooded_0_or_1.mean(), 3))

rad = np.load("../10_radar_like_frames/radar_like_frames_SYNTHETIC.npz")
print("RADAR-LIKE", rad["rain_mm_per_hr"].shape, "(event, 10-min frame, lat, lon)")

b = np.load("../11_second_city/city_B_synthetic_rain.npz")
print("CITY B", b["rain_mm_per_hr"].shape)

k = int(np.argmax(R.reshape(len(R), -1).max(1)))
plt.imshow(R[k], origin="lower", vmin=0, vmax=30, cmap="turbo")
plt.colorbar(label="mm/hr"); plt.title("Heaviest synthetic frame: " + str(rain["time"][k])); plt.show()
