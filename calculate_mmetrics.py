import os
import sys
import geopandas as gpd
import pandas as pd
import momepy
import osmnx as ox
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
from shapely.validation import make_valid
from shapely.errors import TopologicalError, GEOSException
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.font_manager as fm
from shapely.geometry import MultiPolygon, MultiLineString
import matplotlib.patches as mpatches
from shapely.ops import unary_union

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

def query_buildings_data(xmin, ymin, xmax, ymax):
    import duckdb
    con = duckdb.connect(os.path.join(grandparent_dir, 'dev/slums-model-unitac/data/0/data.db'))
    con.install_extension('httpfs')
    con.install_extension('spatial')
    con.load_extension('httpfs')
    con.load_extension('spatial')
    con.execute("SET s3_region='us-west-2'")
    con.execute("SET azure_storage_connection_string = 'DefaultEndpointsProtocol=https;AccountName=overturemapswestus2;AccountKey=;EndpointSuffix=core.windows.net';")
    
    query = f"""
        SELECT *
        FROM buildings
        WHERE bbox.xmin > {xmin}
          AND bbox.xmax < {xmax}
          AND bbox.ymin > {ymin}
          AND bbox.ymax < {ymax};
    """
    
    # Execute query and fetch results directly as a pandas DataFrame
    buildings_df = con.execute(query).df()
    
    if not buildings_df.empty:
        # Convert to GeoDataFrame
        buildings = gpd.GeoDataFrame(
            buildings_df,
            geometry=gpd.GeoSeries.from_wkb(buildings_df.geometry.apply(bytes)),
            crs='EPSG:4326'
        )
        buildings = buildings[['id', 'geometry']]
        buildings = buildings.to_crs("EPSG:3857")
        buildings['class_id'] = 1
        
    con.close()
    return buildings

cities = {
    # 'BelizeCity': {'image_path': os.path.join(parent_dir, 'slums-model-unitac/data/0/sentinel_Gee/BLZ_BelizeCity_2024.tif'),
    #                'labels_path': os.path.join(parent_dir, 'slums-model-unitac/data/SHP/BelizeCity_PS.shp')},
    # 'Belmopan': {'image_path': os.path.join(parent_dir, 'slums-model-unitac/data/0/sentinel_Gee/BLZ_Belmopan_2024.tif'),
    #              'labels_path': os.path.join(parent_dir, 'slums-model-unitac/data/SHP/Belmopan_PS.shp')},
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
    try:
        buildings = query_buildings_data(bounds[0], bounds[1], bounds[2], bounds[3])
        if buildings.empty:
            print(f"No buildings found for {city_name}")
            return None, None
    except Exception as e:
        print(f"Error querying buildings for {city_name}: {str(e)}")
        return None, None

    # Rest of the function remains the same...
    utm_crs = 'EPSG:32616'
    slums_utm = slums.to_crs(utm_crs)
    buildings_utm = buildings.to_crs(utm_crs)
    buildings_utm['uID'] = range(len(buildings_utm))
    
    # Create study area
    study_area = buildings_utm.unary_union.convex_hull.buffer(100)
    
    # Generate tessellation
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
    tessellation['perimeter'] = tessellation.geometry.length
    tessellation['circular_compactness'] = momepy.CircularCompactness(tessellation).series
    tessellation['convexity'] = momepy.Convexity(tessellation).series
    tessellation['orientation'] = momepy.Orientation(tessellation).series
    tessellation['elongation'] = momepy.Elongation(tessellation).series
    tessellation['rectangularity'] = momepy.Rectangularity(tessellation).series
    
    # Create spatial weights matrices for buildings
    print("Creating spatial weights matrices...")
    sw_queen = libpysal.weights.Queen.from_dataframe(
        buildings_utm, 
        ids='uID',
        silence_warnings=True
    )
    
    sw_dist = libpysal.weights.DistanceBand.from_dataframe(
        buildings_utm,
        threshold=100,
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
    buildings_utm['building_adjacency'] = momepy.BuildingAdjacency(
        buildings_utm,
        sw_queen,
        unique_id='uID'
    ).series
    
    buildings_utm['neighbor_distance'] = momepy.NeighborDistance(
        buildings_utm,
        sw_dist,
        'uID'
    ).series
    
    # Coverage metrics
    print("Calculating coverage metrics...")
    tessellation['car'] = momepy.AreaRatio(
        tessellation, 
        buildings_utm, 
        'cell_area',
        'building_area',
        'uID'
    ).series
    
    # Identify slum/non-slum areas
    slums_union = slums_utm.geometry.union_all()
    tessellation['is_slum'] = tessellation.geometry.intersects(slums_union)
    
    # Transfer slum classification to buildings
    buildings_utm = gpd.sjoin(
        buildings_utm,
        tessellation[['geometry', 'is_slum']],
        how='left',
        predicate='within'
    )
    
    buildings_utm['is_slum'] = buildings_utm['is_slum'].fillna(False)
    
    # Create output directory
    output_dir = f'metrics/{city_name}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    print(f"Saving metrics for {city_name}...")
    building_metrics_df = buildings_utm.drop(columns='geometry')
    building_metrics_df.to_csv(os.path.join(output_dir, 'building_metrics.csv'), index=False)
    
    tessellation_metrics_df = tessellation.drop(columns='geometry')
    tessellation_metrics_df.to_csv(os.path.join(output_dir, 'tessellation_metrics.csv'), index=False)
    
    # Save shapefiles
    buildings_utm.to_file(os.path.join(output_dir, 'buildings_with_metrics.gpkg'), driver='GPKG')
    tessellation.to_file(os.path.join(output_dir, 'tessellation_with_metrics.gpkg'), driver='GPKG')
    
    print(f"Metrics saved to {output_dir} directory")
    
    return buildings_utm, tessellation

def process_all_cities(cities_dict):
    """Process all cities in the dictionary"""
    # Load urban areas data once
    urban_areas = gpd.read_file('data/allurban_SICA.geojson')
    
    # Dictionary to store results
    results = {}
    
    for city_name, city_data in cities_dict.items():
        print(f"\nStarting processing of {city_name}...")
        try:
            buildings_result, tessellation_result = process_city(city_name, city_data, urban_areas)
            results[city_name] = {
                'buildings': buildings_result,
                'tessellation': tessellation_result
            }
            print(f"Successfully processed {city_name}")
        except Exception as e:
            print(f"Failed to process {city_name}: {str(e)}")
            results[city_name] = {
                'buildings': None,
                'tessellation': None,
                'error': str(e)
            }
    
    return results

# Run the analysis for all cities
results = process_all_cities(cities)

# Print summary of processing
print("\nProcessing Summary:")
for city_name, result in results.items():
    if result['buildings'] is not None and result['tessellation'] is not None:
        print(f"{city_name}: Successfully processed")
        print(f"  Buildings: {len(result['buildings'])} features")
        print(f"  Tessellation cells: {len(result['tessellation'])} features")
    else:
        print(f"{city_name}: Failed to process")
        if 'error' in result:
            print(f"  Error: {result['error']}")

##################
#### PLOTTING ####
################## 

# distribution of morphometrics by city
file_path = ('metrics/all_cities_slum_morphometrics.csv')
all_cities_df = pd.read_csv(file_path)

# plt.style.use('ggplot')
city_name_map = {
    'Sansalvador_Ps_': 'San Salvador, El Salvador',
    'SantoDomingoDOM': 'Santo Domingo, Dominican Republic',
    'GuatemalaCity': 'Guatemala City, Guatemala',
    'TegucigalpaHND': 'Tegucigalpa, Honduras',
    'SanJoseCRI': 'San Jose, Costa Rica',
    'Panama': 'Panama City, Panama',
    'Belizecity_': 'Belize City, Belize',
    'Managua': 'Managua, Nicaragua',
    'Belmopan_': 'Belmopan, Belize'
}

all_cities_df = all_cities_df[all_cities_df['city'].isin(['SantoDomingoDOM', 'GuatemalaCity', 'TegucigalpaHND','Panama', 'SanJoseCRI','Managua'])]
all_cities_df['city'] = all_cities_df['city'].map(city_name_map)

# Define numeric columns
numeric_cols = ['tessellation_car', 'buildings_wall', 'buildings_adjacency', 'buildings_neighbour_distance']
numeric_cols_names = ['Tesselation CAR', 'Buildings Perimeter Length', 'Buildings Adjecency', 'Distance to Nearest Building']
col_name_map = dict(zip(numeric_cols, numeric_cols_names))

def remove_outliers(group):
    for col in numeric_cols:
        mean = group[col].mean()
        std = group[col].std()
        group = group[(group[col] >= mean - 3*std) & (group[col] <= mean + 3*std)]
    return group

# Remove outliers for each city separately
all_cities_df = all_cities_df.groupby('city').apply(remove_outliers).reset_index(drop=True)

# Standardize the numeric columns
scaler = MinMaxScaler()
all_cities_df[numeric_cols] = scaler.fit_transform(all_cities_df[numeric_cols])

df_melted = all_cities_df.melt(id_vars=['city'], value_vars=numeric_cols, var_name='Variable', value_name='Value')
df_melted['Variable'] = df_melted['Variable'].map(col_name_map)

# Set up the font
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Garamond'] + plt.rcParams['font.serif']

# Set up the style manually
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.color'] = '#E6E6E6'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Create the plot
plt.figure(figsize=(20, 12))
ax = sns.boxplot(x='Variable', y='Value', hue='city', data=df_melted,
                 palette="Set2", whis=(10, 90))

# Color the outliers the same as their respective bars
for i, artist in enumerate(ax.artists):
    col = artist.get_facecolor()
    for j in range(i*6, i*6+6):
        line = ax.lines[j]
        line.set_color(col)
        line.set_mfc(col)
        line.set_mec(col)

# Customize the plot
plt.title('Standardized Distributions of Building Morphometrics by City', fontsize=28, pad=20)
plt.xlabel('Morphometric', fontsize=20, labelpad=15)
plt.ylabel('Standardized Value', fontsize=20, labelpad=15)
plt.xticks(rotation=45, ha='right', fontsize=18)
ax.set_xticklabels([label.get_text().replace('_', ' ').title() for label in ax.get_xticklabels()])
plt.yticks(fontsize=14)
plt.legend(title='City', title_fontsize='24', fontsize='17', bbox_to_anchor=(1, 1), loc='upper left')
plt.tight_layout()
plt.show()




# #### MAPPING ####
# import geopandas as gpd
# import matplotlib.pyplot as plt
# import numpy as np
# from shapely.geometry import box, Point
# import os
# import duckdb
# from pyproj import Transformer

# def create_square_bbox(lat, lon, size_meters):
#     point = Point(lon, lat)
#     transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
#     x, y = transformer.transform(lon, lat)
#     half_size = size_meters / 2
#     bbox = box(x - half_size, y - half_size, x + half_size, y + half_size)
#     transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
#     minx, miny = transformer.transform(bbox.bounds[0], bbox.bounds[1])
#     maxx, maxy = transformer.transform(bbox.bounds[2], bbox.bounds[3])
#     return minx, miny, maxx, maxy

# def plot_buildings_and_precarious_areas(ax, lat, lon, size_meters, city_name, grandparent_dir):
#     xmin, ymin, xmax, ymax = create_square_bbox(lat, lon, size_meters)
    
#     con = duckdb.connect(os.path.join(parent_dir, 'data/0/data.db'))
#     con.install_extension('httpfs')
#     con.install_extension('spatial')
#     con.load_extension('httpfs')
#     con.load_extension('spatial')
#     con.execute("SET s3_region='us-west-2'")
#     con.execute("SET azure_storage_connection_string = 'DefaultEndpointsProtocol=https;AccountName=overturemapswestus2;AccountKey=;EndpointSuffix=core.windows.net';")
    
#     query = f"""
#         SELECT *
#         FROM buildings
#         WHERE bbox.xmin > {xmin}
#           AND bbox.xmax < {xmax}
#           AND bbox.ymin > {ymin}
#           AND bbox.ymax < {ymax};
#     """
#     buildings_df = gpd.read_postgis(query, con, geom_col='geometry', crs='EPSG:4326')
    
#     city_name_map = {
#         'Sansalvador_Ps_': 'San Salvador, El Salvador',
#         'SantoDomingo': 'Santo Domingo, Dominican Republic',
#         'GuatemalaCity': 'Guatemala City, Guatemala',
#         'Tegucigalpa': 'Tegucigalpa, Honduras',
#         'SanJoseCRI': 'San Jose, Costa Rica',
#         'Panama': 'Panama City, Panama',
#         'BelizeCity': 'Belize City, Belize (excluded from the study)',
#         'Managua': 'Managua, Nicaragua',
#         'Belmopan_': 'Belmopan, Belize'
#     }
    
#     cities = {
#         'Tegucigalpa': os.path.join(parent_dir, 'data/SHP/Tegucigalpa_PS.shp'),
#         'SantoDomingo': os.path.join(parent_dir, 'data/0/SantoDomingo3857_buffered.geojson'),
#         'GuatemalaCity': os.path.join(parent_dir, 'data/SHP/Guatemala_PS.shp'),
#         'Managua': os.path.join(parent_dir, 'data/SHP/Managua_PS.shp'),
#         'Panama': os.path.join(parent_dir, 'data/SHP/Panama_PS.shp'),
#         'BelizeCity': os.path.join(parent_dir, 'data/SHP/BelizeCity_PS.shp'),
#         'SanJoseCRI': os.path.join(parent_dir, 'data/SHP/SanJose_PS.shp')
#     }
        
#     precarious_areas = gpd.read_file(cities[city_name])
    
#     buildings_df = buildings_df.to_crs('EPSG:3857')
#     precarious_areas = precarious_areas.to_crs('EPSG:3857')
    
#     bbox = box(xmin, ymin, xmax, ymax)
#     bbox_gdf = gpd.GeoDataFrame({'geometry': [bbox]}, crs='EPSG:4326').to_crs('EPSG:3857')
    
#     precarious_areas = gpd.clip(precarious_areas, bbox_gdf)
    
#     unified_precarious_area = precarious_areas.union_all()
#     unified_precarious_gdf = gpd.GeoDataFrame({'geometry': [unified_precarious_area]}, crs='EPSG:3857')
    
#     ax.set_facecolor('white')
    
#     buildings_df.plot(ax=ax, edgecolor='none', facecolor='black', linewidth=0)
    
#     # Plot precarious areas with faint fill and outline
#     unified_precarious_gdf.plot(ax=ax, facecolor='red', edgecolor='red', alpha=0.05, linewidth=2)
#     unified_precarious_gdf.boundary.plot(ax=ax, color='red', linewidth=2)
    
#     ax.set_xlim(bbox_gdf.total_bounds[0], bbox_gdf.total_bounds[2])
#     ax.set_ylim(bbox_gdf.total_bounds[1], bbox_gdf.total_bounds[3])
    
#     ax.set_axis_off()
    
#     cleaned_city_name = city_name_map.get(city_name, city_name)
#     ax.set_title(f'{cleaned_city_name}', fontsize=14)

# def plot_multiple_areas(coordinates_list, size_meters, grandparent_dir):
#     n = len(coordinates_list)
#     cols = int(np.ceil(np.sqrt(n)))
#     rows = int(np.ceil(n / cols))
    
#     fig, axs = plt.subplots(rows, cols, figsize=(5*cols, 5*rows), squeeze=False)
#     fig.suptitle('Building Footprints and Precarious Areas in Different Cities', fontsize=16)
    
#     for i, (lat, lon, city_name) in enumerate(coordinates_list):
#         row = i // cols
#         col = i % cols
#         ax = axs[row, col]
#         plot_buildings_and_precarious_areas(ax, lat, lon, size_meters, city_name, grandparent_dir)
    
#     # Hide any unused subplots
#     for i in range(n, rows*cols):
#         row = i // cols
#         col = i % cols
#         axs[row, col].axis('off')
    
#     plt.tight_layout()
#     plt.show()

# # Example usage
# coordinates_list = [
#     (14.643041208660227, -90.52696617369627, 'GuatemalaCity'),
#     (14.09833557316007, -87.24716065494343, 'Tegucigalpa'),
#     (18.506793321891678, -69.89322847545206, 'SantoDomingo'),
#     (12.153548471297961, -86.25461143585959, 'Managua'),
#     (8.925055642079421, -79.62752568376733, 'Panama'),
#     (9.946122438272413, -84.08819439919527, 'SanJoseCRI'),
# ]

# size_meters = 1500

# plot_multiple_areas(coordinates_list, size_meters, grandparent_dir)