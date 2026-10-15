# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: bicycle-network:default (3.14.8)
#     language: python
#     name: python3
# ---

# %%
import geopandas as gpd

# %%
cph = gpd.read_file("../../data/afstemningsomraader2025_CPH_FRB.geojson")

# %%
cph.info(verbose=True)

# %%
cph.head()

# %% [markdown]
# now merge polygons to get one polygon. then make hexagonal point grid

# %%
print(cph.crs.to_string())

# %%
cph.geometry.head()

# %%
cph.geom_type.unique()

# %%
cph.plot()

# %%
cph.explore()

# %%
cph.loc[27, 'geometry']

# %%
cph.crs

# %% [markdown]
# "To obtain area measurements, first make sure your `GeoDataFrame` is in a *projected* CRS. GeoPandas will issue a warning if you try to do distance or area computations in a geographic CRS." (from GDS lec 2)

# %%
cph.area.head()

# %% [markdown]
# We get a warning because our CRS is geographic, not projected. Distances would be measured in degrees instead of e.g. metres. Change the geometries to a projected CRS:

# %%
cph_proj = cph.to_crs(epsg=25832)
cph_proj.plot()

# %%
cph.plot()

# %%
cph_proj.area.head()

# %%
# Find centroids
centroids = cph_proj.centroid
centroids.head()

# %%
centroids.plot()

# %%
import matplotlib.pyplot as plt

# %%
buf = centroids.buffer(500)
f, ax = plt.subplots(1)
buf.plot(ax=ax, linewidth=0)

# %%
import contextily as cx

# %%
f, ax = plt.subplots(1, figsize=(9, 9))
cph.plot(alpha=0.6, ax=ax)
cx.add_basemap(
    ax, 
    crs=cph.crs,
    source=cx.providers.Esri.WorldImagery
)
ax.set_axis_off()

# %%
import shapely

# %%
# Merge polygons

polygon = shapely.unary_union(cph.geometry)
polygon

# %%
polygon_proj = shapely.unary_union(cph_proj.geometry)
polygon_proj

# %%
