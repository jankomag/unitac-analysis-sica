import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
import seaborn as sns
import ee
import os
import sys
import geemap
from shapely.geometry import box, Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely import make_valid, simplify, wkt
from tqdm import tqdm
from rasterio.transform import from_bounds
import pandas as pd
from rasterio.features import geometry_mask
import folium
import matplotlib.pyplot as plt
from shapely.ops import unary_union
from shapely.geometry import MultiPolygon, box, Polygon
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import re
from matplotlib import rcParams
import contextily as cx
from matplotlib_scalebar.scalebar import ScaleBar
from pyproj import CRS
from shapely.geometry import box, Point



import geopandas as gpd
import pandas as pd
import ee
from tqdm import tqdm
from shapely.geometry import Polygon, MultiPolygon
import time
from functools import wraps
import random

import geopandas as gpd
import pandas as pd
import ee
from tqdm import tqdm
from shapely.geometry import Polygon, MultiPolygon
import time
from functools import wraps
import random

def retry_with_backoff(retries=3, backoff_in_seconds=1):
    """Retry decorator with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if x == retries:
                        raise e
                    sleep = (backoff_in_seconds * 2 ** x + 
                            random.uniform(0, 1))
                    time.sleep(sleep)
                    x += 1
        return wrapper
    return decorator

def load_slum_polygons(city_data):
    """Load slum polygons for a given city."""
    slums = gpd.read_file(city_data['labels_path'])
    if slums.crs != 'EPSG:4326':
        slums = slums.to_crs('EPSG:4326')
    return slums

def get_intersecting_urban_areas(slums_gdf, urban_areas_gdf):
    """Find urban areas that intersect with the slum polygons."""
    if urban_areas_gdf.crs != 'EPSG:4326':
        urban_areas_gdf = urban_areas_gdf.to_crs('EPSG:4326')
    if slums_gdf.crs != 'EPSG:4326':
        slums_gdf = slums_gdf.to_crs('EPSG:4326')
    
    intersecting = urban_areas_gdf[urban_areas_gdf.intersects(slums_gdf.unary_union)]
    return intersecting

def geometry_to_ee(geometry):
    """Convert Shapely geometry to Earth Engine geometry."""
    if isinstance(geometry, Polygon):
        exterior_coords = [[coord[0], coord[1]] for coord in geometry.exterior.coords]
        interior_coords = []
        for interior in geometry.interiors:
            interior_coords.append([[coord[0], coord[1]] for coord in interior.coords])
        all_coords = [exterior_coords] + interior_coords
        return ee.Geometry.Polygon(all_coords)
    
    elif isinstance(geometry, MultiPolygon):
        polygons = []
        for poly in geometry.geoms:
            exterior_coords = [[coord[0], coord[1]] for coord in poly.exterior.coords]
            interior_coords = []
            for interior in poly.interiors:
                interior_coords.append([[coord[0], coord[1]] for coord in interior.coords])
            polygons.append([exterior_coords] + interior_coords)
        return ee.Geometry.MultiPolygon(polygons)
    else:
        raise ValueError(f"Unsupported geometry type: {type(geometry)}")

@retry_with_backoff(retries=5, backoff_in_seconds=2)
def calculate_population(geometry, population_image, scale, band_name):
    """Calculate population within a geometry using Earth Engine with retries."""
    ee_geometry = geometry_to_ee(geometry)
    
    population = population_image.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=ee_geometry,
        scale=scale,
        maxPixels=1e9
    ).get(band_name).getInfo()
    
    return population if population is not None else 0

def process_geometries_batch(geometries, population_image, scale, band_name, batch_size=5):
    """Process geometries in batches to avoid overwhelming the connection."""
    total = 0
    for i in range(0, len(geometries), batch_size):
        batch = geometries[i:i + batch_size]
        batch_total = sum(calculate_population(geom, population_image, scale, band_name) 
                         for geom in batch)
        total += batch_total
        # Small delay between batches
        time.sleep(1)
    return total

def analyze_populations(cities, urban_areas):
    """
    Analyze populations using both HRSL and GHS datasets with their correct scales and band names.
    Returns a DataFrame with population estimates for each city.
    """
    # Initialize Earth Engine datasets with their respective scales and band names
    datasets = {
        "HRSL": {
            "image": ee.ImageCollection("projects/sat-io/open-datasets/hrsl/hrslpop").mosaic(),
            "scale": 30,
            "band_name": "b1"
        },
        "GHS": {
            "image": ee.Image('JRC/GHSL/P2023A/GHS_POP/2015'),
            "scale": 100,
            "band_name": "population_count"
        }
    }
    
    results = []
    
    for city_name, city_data in tqdm(cities.items(), desc="Processing cities"):
        print(f"\nProcessing {city_name}")
        
        # Load slum polygons
        slums_gdf = load_slum_polygons(city_data)
        
        # Find intersecting urban areas
        city_urban_areas = get_intersecting_urban_areas(slums_gdf, urban_areas)
        has_urban_data = not city_urban_areas.empty
        
        # Calculate populations using both datasets
        for dataset_name, dataset_info in datasets.items():
            try:
                population_image = dataset_info["image"]
                scale = dataset_info["scale"]
                band_name = dataset_info["band_name"]
                
                print(f"Calculating {dataset_name} populations...")
                
                # Calculate slum population in batches
                slum_pop = process_geometries_batch(
                    list(slums_gdf.geometry), 
                    population_image, 
                    scale,
                    band_name
                )
                
                # Calculate urban population if available
                urban_pop = 0
                if has_urban_data:
                    urban_pop = process_geometries_batch(
                        list(city_urban_areas.geometry), 
                        population_image, 
                        scale,
                        band_name
                    )
                
                result = {
                    'city': city_name,
                    'dataset': dataset_name,
                    'scale': scale,
                    'slum_population': slum_pop,
                    'urban_population': urban_pop if has_urban_data else None,
                    'slum_proportion': (slum_pop / urban_pop if urban_pop > 0 else None) 
                                     if has_urban_data else None
                }
                
                results.append(result)
                
                print(f"{dataset_name} Results for {city_name}:")
                print(f"Scale: {scale}m")
                print(f"Slum Population: {slum_pop:,.0f}")
                if has_urban_data:
                    print(f"Urban Population: {urban_pop:,.0f}")
                    if urban_pop > 0:
                        print(f"Slum Proportion: {(slum_pop/urban_pop)*100:.1f}%")
                
                # Add delay between datasets
                time.sleep(2)
                
            except Exception as e:
                print(f"Error processing {city_name} with {dataset_name}: {str(e)}")
                continue
    
    # Create DataFrame and save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('data/population_analysis.csv', index=False)
    return results_df

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Initialize Earth Engine
ee.Authenticate()
ee.Initialize(project='ee-yankomagn')

cities = {
    'BelizeCity': {'image_path': os.path.join(parent_dir, 'slums-model-unitac/data/0/sentinel_Gee/BLZ_BelizeCity_2024.tif'),
                'labels_path': os.path.join(parent_dir, 'slums-model-unitac/data/SHP/BelizeCity_PS.shp')},
    'Belmopan': {'image_path': os.path.join(parent_dir, 'slums-model-unitac/data/0/sentinel_Gee/BLZ_Belmopan_2024.tif'),
                'labels_path': os.path.join(parent_dir, 'slums-model-unitac/data/SHP/Belmopan_PS.shp')},
    'Tegucigalpa': {
        'image_path': os.path.join(parent_dir, 'slums-model-unitac/data/0/sentinel_Gee/HND_Comayaguela_2023.tif'),
        'labels_path': os.path.join(parent_dir, 'slums-model-unitac/data/SHP/Tegucigalpa_PS.shp'),
        'use_augmentation': False
    },
    'SantoDomingo': {
        'image_path': os.path.join(parent_dir, 'slums-model-unitac/data/0/sentinel_Gee/DOM_Los_Minas_2024.tif'),
        'labels_path': os.path.join(parent_dir, 'slums-model-unitac/data/0/SantoDomingo3857_buffered.geojson'),
    },
    'GuatemalaCity': {
        'image_path': os.path.join(parent_dir, 'slums-model-unitac/data/0/sentinel_Gee/GTM_Guatemala_2024.tif'),
        'labels_path': os.path.join(parent_dir, 'slums-model-unitac/data/SHP/Guatemala_PS.shp'),
    },
    'Managua': {
        'image_path': os.path.join(parent_dir, 'slums-model-unitac/data/0/sentinel_Gee/NIC_Tipitapa_2023.tif'),
        'labels_path': os.path.join(parent_dir, 'slums-model-unitac/data/SHP/Managua_PS.shp'),
        'use_augmentation': False
    },
    'Panama': {
        'image_path': os.path.join(parent_dir, 'slums-model-unitac/data/0/sentinel_Gee/PAN_Panama_2024.tif'),
        'labels_path': os.path.join(parent_dir, 'slums-model-unitac/data/SHP/Panama_PS.shp'),
        'use_augmentation': False
    },
    'SanSalvador_PS': {
        'image_path': os.path.join(parent_dir, 'slums-model-unitac/data/0/sentinel_Gee/SLV_SanSalvador_2024.tif'),
        'labels_path': os.path.join(parent_dir, 'slums-model-unitac/data/SHP/SanSalvador_PS_lotifi_ilegal.shp'),
    },
    'SanJoseCRI': {
        'image_path': os.path.join(parent_dir, 'slums-model-unitac/data/0/sentinel_Gee/CRI_San_Jose_2023.tif'),
        'labels_path': os.path.join(parent_dir, 'slums-model-unitac/data/SHP/SanJose_PS.shp'),
        'use_augmentation': False
    }
}

# Load urban areas
urban_areas = gpd.read_file('data/allurban_SICA.geojson')

# Run analysis
results = analyze_populations(cities, urban_areas)

# Display summary
print("\nAnalysis Summary:")
summary = pd.pivot_table(
    results,
    index='city',
    columns='dataset',
    values=['slum_population', 'urban_population', 'slum_proportion']
)
summary = summary.round({'slum_population': 0, 'urban_population': 0, 'slum_proportion': 3})
print(summary)