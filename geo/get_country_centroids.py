import geopandas as gpd
import requests
import io
import pandas as pd

def get_country_centroids():
    url = "https://d2ad6b4ur7yvpq.cloudfront.net/naturalearth-3.3.0/ne_50m_admin_0_countries.geojson"
    r = requests.get(url, timeout=60)
    world = gpd.read_file(io.BytesIO(r.content))
    world_m = world.to_crs(epsg=3857)
    world['centroid'] = world_m.geometry.centroid.to_crs(epsg=4326)
    centroids = {
        iso: (pt.y, pt.x)
        for iso, pt in zip(world['iso_a3'], world['centroid'])
        if pd.notna(iso)
    }
    return centroids

if __name__ == "__main__":
    c = get_country_centroids()
    print(c['USA'], c['CHN'])
