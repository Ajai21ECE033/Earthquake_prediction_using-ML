import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import mplcursors

# Load the earthquake data
path = 'https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.csv'
data = pd.read_csv(path)

# Create a GeoDataFrame from the latitude and longitude columns
gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.longitude, data.latitude))

# Plot the world map
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
ax = world.plot(figsize=(15, 10), color='lightgray', edgecolor='black')

# Plot the earthquake points on top of the world map
gdf.plot(ax=ax, color='red', markersize=data['mag'] ** 2, alpha=0.7, label='Magnitude')

# Add annotations on hover
cursor = mplcursors.cursor(hover=True)

def on_hover(sel):
    index = sel.target.index
    place = data['place'][index]
    magnitude = data['mag'][index]
    time = data['time'][index]
    text = f"Place: {place}\nMagnitude: {magnitude}\nTime: {time}"
    sel.annotation.set(text=text, position=(data['longitude'][index], data['latitude'][index]))
    sel.annotation.arrow_patch.set(arrowstyle='simple', fc="white", alpha=0.5)
    sel.annotation.get_bbox_patch().set(fc="white", alpha=1)

cursor.connect("add", on_hover)

plt.title('Earthquake Data on World Map')
plt.legend()
plt.show()
