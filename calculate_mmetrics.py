import os
import sys
import geopandas as gpd
import pandas as pd
import momepy
import osmnx as ox
import shapely
from shapely.geometry import box
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import MultiPolygon
import libpysal
from shapely.validation import make_valid
from shapely.errors import TopologicalError, GEOSException
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.font_manager as fm
from shapely.geometry import MultiPolygon, MultiLineString
import networkx as nx
from shapely.geometry import box
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import MultiPolygon
import libpysal
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.validation import make_valid
from shapely.errors import TopologicalError, GEOSException
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.font_manager as fm
from shapely.geometry import MultiPolygon, MultiLineString
import matplotlib.patches as mpatches
import math
from shapely.ops import unary_union
from multiprocessing import freeze_support

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)

os.makedirs('metrics', exist_ok=True)

def get_utm_crs(lat, lon):
    """
    Calculate the EPSG code for the UTM CRS based on lat/lon coordinates
    """
    utm_band = str(int((lon + 180) / 6) + 1)
    if len(utm_band) == 1:
        utm_band = '0' + utm_band
        
    if lat >= 0:
        epsg_code = '326' + utm_band
    else:
        epsg_code = '327' + utm_band
    
    return f'EPSG:{epsg_code}'

# Simple direct calculation using matching uIDs
def query_buildings_data(xmin, ymin, xmax, ymax, use_cache=True):
    """Query buildings using OvertureMaestro with caching and progress tracking"""
    import overturemaestro as om
    from shapely.geometry import box
    
    # Create bounding box
    bbox = box(xmin, ymin, xmax, ymax)
    
    try:
        # Get buildings data with additional options
        buildings = om.convert_geometry_to_geodataframe(
            theme="buildings",
            type="building",
            geometry_filter=bbox,
            ignore_cache=not use_cache,  # Use cache by default
            verbosity_mode="transient",  # Show progress but clean up after
            working_directory="cache/overture"  # Store cache in specific directory
        )
        
        if buildings.empty:
            print("No buildings found in the specified area")
            return None
        
        # You could add additional filters here if needed
        # Reset the index to make 'id' a regular column
        buildings = buildings.reset_index()
        
        # Now we can select id and geometry
        buildings = buildings[['id', 'geometry']]
        
        buildings = buildings.to_crs("EPSG:3857")
        buildings['class_id'] = 1
        
        return buildings
        
    except Exception as e:
        print(f"Error querying buildings: {str(e)}")
        return None

def query_roads_data(xmin, ymin, xmax, ymax):
    """Query road network using OvertureMaestro"""
    import overturemaestro as om
    from shapely.geometry import box
    
    # Create bounding box
    bbox = box(xmin, ymin, xmax, ymax)
    
    try:
        # Get roads data with filter for road subtype only
        roads = om.convert_geometry_to_geodataframe(
            theme="transportation",
            type="segment",
            geometry_filter=bbox,
            pyarrow_filter=[[
                ("subtype", "=", "road"),  # Only get road segments
                # Exclude service roads, paths, steps, etc if you want only main roads
                ("class", "in", [
                    "motorway", "trunk", "primary", "secondary", 
                    "tertiary", "residential", "unclassified"
                ])
            ]],
            verbosity_mode="silent"
        )
        
        if roads.empty:
            print("No roads found in the specified area")
            return None
        
        # Keep essential columns
        roads = roads.reset_index()
        roads = roads[[
            'id', 'geometry', 'class', 'subclass',
            'names',  # Keep names for street names
            'speed_limits'  # Keep speed limits if available
        ]]
        roads = roads.to_crs("EPSG:3857")
        
        return roads
        
    except Exception as e:
        print(f"Error querying roads: {str(e)}")
        return None

cities = {
    # 'BelizeCity': {'image_path': os.path.join(parent_dir, 'slums-model-unitac/data/0/sentinel_Gee/BLZ_BelizeCity_2024.tif'),
    #                'labels_path': os.path.join(parent_dir, 'slums-model-unitac/data/SHP/BelizeCity_PS.shp')},
    # 'Belmopan': {'image_path': os.path.join(parent_dir, 'slums-model-unitac/data/0/sentinel_Gee/BLZ_Belmopan_2024.tif'),
    #              'labels_patha': os.path.join(parent_dir, 'slums-model-unitac/data/SHP/Belmopan_PS.shp')},
    # 'Tegucigalpa': {
    #     'image_path': os.path.join(parent_dir, 'slums-model-unitac/data/0/sentinel_Gee/HND_Comayaguela_2023.tif'),
    #     'labels_path': os.path.join(parent_dir, 'slums-model-unitac/data/SHP/Tegucigalpa_PS.shp'),
    #     'use_augmentation': False
    # },
    # 'SantoDomingo': {
    #     'image_path': os.path.join(parent_dir, 'slums-model-unitac/data/0/sentinel_Gee/DOM_Los_Minas_2024.tif'),
    #     'labels_path': os.path.join(parent_dir, 'slums-model-unitac/data/0/SantoDomingo3857_buffered.geojson'),
    # },
    # 'GuatemalaCity': {
    #     'image_path': os.path.join(parent_dir, 'slums-model-unitac/data/0/sentinel_Gee/GTM_Guatemala_2024.tif'),
    #     'labels_path': os.path.join(parent_dir, 'slums-model-unitac/data/SHP/Guatemala_PS.shp'),
    # },
    # 'Managua': {
    #     'image_path': os.path.join(parent_dir, 'slums-model-unitac/data/0/sentinel_Gee/NIC_Tipitapa_2023.tif'),
    #     'labels_path': os.path.join(parent_dir, 'slums-model-unitac/data/SHP/Managua_PS.shp'),
    #     'use_augmentation': False
    # },
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

def process_city(city_name, city_data, urban_areas):
    """Process a city with comprehensive metrics for slum/non-slum comparison"""
    print(f"\nProcessing {city_name}...")
    
    # Load slums data
    slums = gpd.read_file(city_data['labels_path'])
    slums = slums.to_crs(epsg=4326)
    
    # Use the provided urban areas data
    urban_areas = urban_areas.to_crs(epsg=4326)
    
    # Find urban areas that intersect with slums
    slums_union = unary_union(slums.geometry)
    intersecting_urban_areas = urban_areas[urban_areas.geometry.intersects(slums_union)]
    
    # Get bounds either from intersection or directly from slums
    if len(intersecting_urban_areas) > 0:
        print(f"Found {len(intersecting_urban_areas)} intersecting urban areas")
        intersecting_urban_union = unary_union(intersecting_urban_areas.geometry)
        bounds = intersecting_urban_union.bounds
    else:
        print(f"No intersecting urban areas found for {city_name}, using slums extent")
        # Add a buffer around slums bounds to ensure we capture surrounding context
        slums_bounds = slums_union.bounds
        buffer_size = 0.01  # ~1km at equator
        bounds = (
            slums_bounds[0] - buffer_size,  # xmin
            slums_bounds[1] - buffer_size,  # ymin
            slums_bounds[2] + buffer_size,  # xmax
            slums_bounds[3] + buffer_size   # ymax
        )
    
    # Query buildings using the bounds
    buildings = query_buildings_data(bounds[0], bounds[1], bounds[2], bounds[3])
    
    if buildings.empty:
        print(f"No buildings found for {city_name}")
        return None
    
    utm_crs = 'EPSG:32616'
    slums_utm = slums.to_crs(utm_crs)
    buildings_utm = buildings.to_crs(utm_crs)
    buildings_utm['uID'] = range(len(buildings_utm))
    
    # Create study area
    study_area = buildings_utm.unary_union.convex_hull.buffer(100)
    
    # Generate tessellation
    print("Generating tessellation...")
    tess = momepy.Tessellation(
        buildings_utm,
        'uID',
        limit=study_area,
        shrink=0.4,
        segment=0.5,
    )
    tessellation = tess.tessellation
    
    # Calculate tessellation metrics
    print("Calculating tessellation metrics...")
    tessellation['cell_area'] = tessellation.geometry.area
    tessellation['cell_perimeter'] = tessellation.geometry.length
    tessellation['cell_circular_compactness'] = momepy.CircularCompactness(tessellation).series
    tessellation['cell_convexity'] = momepy.Convexity(tessellation).series
    tessellation['cell_orientation'] = momepy.Orientation(tessellation).series
    tessellation['cell_elongation'] = momepy.Elongation(tessellation).series
    tessellation['cell_rectangularity'] = momepy.Rectangularity(tessellation).series
    
    # Create spatial weights matrices for buildings
    print("Creating spatial weights matrices...")
    sw_distance100 = libpysal.weights.DistanceBand.from_dataframe(
        buildings_utm,
        threshold=100,
        ids='uID',
        silence_warnings=True
    )
    sw_distance1 = libpysal.weights.DistanceBand.from_dataframe(
        buildings_utm,
        threshold=1,
        ids='uID',
        silence_warnings=True
    )
    
    # Calculate building metrics
    print("Calculating building metrics...")
    buildings_utm['building_area'] = buildings_utm.geometry.area
    buildings_utm['perimeter'] = buildings_utm.geometry.length
    buildings_utm['longest_axis'] = momepy.LongestAxisLength(buildings_utm).series
    buildings_utm['elongation'] = momepy.Elongation(buildings_utm).series
    buildings_utm['orientation'] = momepy.Orientation(buildings_utm).series
    buildings_utm['corners'] = momepy.Corners(buildings_utm).series
    buildings_utm['fractal_dimension'] = momepy.FractalDimension(buildings_utm).series    
    buildings_utm['squareness'] = momepy.Squareness(buildings_utm).series
    buildings_utm['circular_compactness'] = momepy.CircularCompactness(buildings_utm).series
    buildings_utm['convexity'] = momepy.Convexity(buildings_utm).series
    buildings_utm['rectangularity'] = momepy.Rectangularity(buildings_utm).series
    
    # Adjacency and distance metrics
    print("Calculating adjacency and distance metrics...")
    
    # Calculate building adjacency using distance weights
    buildings_utm['building_adjacency'] = momepy.BuildingAdjacency(
        buildings_utm,
        sw_distance1,
        unique_id='uID'
    ).series
    buildings_utm['building_adjacency'] = buildings_utm['building_adjacency'].fillna(0)
    
    buildings_utm['neighbor_distance'] = momepy.NeighborDistance(
        buildings_utm,
        sw_distance100,
        'uID'
    ).series
    
    # Calculate building alignment with neighbors
    buildings_utm['alignment'] = momepy.Alignment(
        buildings_utm,
        sw_distance100, 
        'uID',
        'orientation'
    ).series

    # Coverage metrics
    print("Calculating coverage metrics...")
    tessellation['coverage_ratio'] = (
        buildings_utm.set_index('uID')['building_area'] / 
        tessellation.set_index('uID')['cell_area']
    ).reindex(tessellation.index).fillna(0)
    
    print(f"NAs in coverage ratio: {tessellation['coverage_ratio'].isna().sum()}")
    
    # Identify slum/non-slum areas
    slums_union = slums_utm.geometry.union_all()
    buildings_utm['is_slum'] = buildings_utm.geometry.intersects(slums_union)
    
    # Transfer tessellation metrics to buildings
    # First join slum classification and coverage ratio
    buildings_utm = buildings_utm.merge(
        tessellation[[
            'uID',  # Include uID column in the selection
            'coverage_ratio',
            'cell_area',
            'cell_perimeter',
            'cell_circular_compactness',
            'cell_convexity',
            'cell_orientation',
            'cell_elongation',
            'cell_rectangularity'
        ]],
        on='uID',  # Specify the column to join on
        how='left'
    )
    # Fill NA values\
    buildings_utm['is_slum'] = buildings_utm['is_slum'].fillna(False)
    for col in buildings_utm.columns:
        if col.startswith('cell_'):
            buildings_utm[col] = buildings_utm[col].fillna(-1)
    
    # Create output directory
    output_dir = f'metrics/{city_name}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    print(f"Saving metrics for {city_name}...")
    # Save all metrics in one CSV file
    metrics_df = buildings_utm.drop(columns=['geometry'])
    metrics_df.to_csv(os.path.join(output_dir, 'combined_metrics.csv'), index=False)
    
    # Save shapefile of buildings with all metrics
    buildings_utm.to_file(os.path.join(output_dir, 'buildings_with_metrics.gpkg'), driver='GPKG')
    tessellation.to_file(os.path.join(output_dir, 'tessellation_with_metrics.gpkg'), driver='GPKG')
    
    print(f"Metrics saved to {output_dir} directory")
    
    return buildings_utm

def process_all_cities(cities_dict):
    """Process all cities in the dictionary"""
    # Load urban areas data once
    urban_areas = gpd.read_file('data/allurban_SICA.geojson')
    
    # Dictionary to store results
    results = {}
    
    for city_name, city_data in cities_dict.items():
        print(f"\nStarting processing of {city_name}...")
        try:
            buildings_metrics = process_city(city_name, city_data, urban_areas)
            
            if buildings_metrics is not None:
                results[city_name] = {
                    'buildings': buildings_metrics,
                    'success': True
                }
                print(f"Successfully processed {city_name}")
            else:
                results[city_name] = {
                    'buildings': None,
                    'success': False,
                    'error': "No buildings data found"
                }
                print(f"No buildings data found for {city_name}")
                
        except Exception as e:
            print(f"Failed to process {city_name}: {str(e)}")
            results[city_name] = {
                'buildings': None,
                'success': False,
                'error': str(e)
            }
    
    return results

# Run the analysis for all cities
if __name__ == '__main__':
    # Add this line at the start of your main script
    from multiprocessing import freeze_support
    freeze_support()
    
    # Then your existing code
    results = process_all_cities(cities)