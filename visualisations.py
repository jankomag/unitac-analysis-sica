import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)

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
    }
    # 'Panama': {
    #     'image_path': os.path.join(parent_dir, 'slums-model-unitac/data/0/sentinel_Gee/PAN_Panama_2024.tif'),
    #     'labels_path': os.path.join(parent_dir, 'slums-model-unitac/data/SHP/Panama_PS.shp'),
    #     'use_augmentation': False
    # },
    # 'SanSalvador_PS': {
    #     'image_path': os.path.join(parent_dir, 'slums-model-unitac/data/0/sentinel_Gee/SLV_SanSalvador_2024.tif'),
    #     'labels_path': os.path.join(parent_dir, 'slums-model-unitac/data/SHP/SanSalvador_PS_lotifi_ilegal.shp'),
    # },
    # 'SanJoseCRI': {
    #     'image_path': os.path.join(parent_dir, 'slums-model-unitac/data/0/sentinel_Gee/CRI_San_Jose_2023.tif'),
    #     'labels_path': os.path.join(parent_dir, 'slums-model-unitac/data/SHP/SanJose_PS.shp'),
    #     'use_augmentation': False
    # }
}

import os
import pandas as pd
import glob

def load_combined_metrics(metrics_dir='metrics'):
    """
    Load metrics from all cities into a single DataFrame.
    
    Parameters:
    -----------
    metrics_dir : str
        Base directory containing city metric subdirectories
        
    Returns:
    --------
    pandas.DataFrame
        Combined DataFrame with metrics from all cities
    """
    # List all city directories
    city_dirs = glob.glob(os.path.join(metrics_dir, '*'))
    
    # Initialize empty list to store DataFrames
    dfs = []
    
    # Process each city directory
    for city_dir in city_dirs:
        city_name = os.path.basename(city_dir)
        metrics_file = os.path.join(city_dir, 'combined_metrics.csv')
        
        if os.path.exists(metrics_file):
            # Read the CSV file
            try:
                df = pd.read_csv(metrics_file)
                # Add city name as a column
                df['city'] = city_name
                dfs.append(df)
                print(f"Successfully loaded metrics for {city_name}")
            except Exception as e:
                print(f"Error loading metrics for {city_name}: {str(e)}")
        else:
            print(f"No metrics file found for {city_name}")
    
    if not dfs:
        raise ValueError("No metric files were successfully loaded")
    
    # Combine all DataFrames
    combined_df = pd.concat(dfs, ignore_index=True)
    
    print(f"\nCombined dataset summary:")
    print(f"Total number of buildings: {len(combined_df)}")
    print(f"Number of cities: {combined_df['city'].nunique()}")
    print(f"Cities included: {', '.join(combined_df['city'].unique())}")
    
    return combined_df

# Load the combined metrics
pd.set_option('display.max_columns', None)
combined_metrics = load_combined_metrics()

def create_standardized_boxplots(combined_metrics, columns_to_visualize):
    """
    Create standardized boxplots for all metrics comparing slum vs non-slum areas.
    """
    # Filter columns
    df = combined_metrics[columns_to_visualize + ['is_slum', 'city']].copy()
    
    # Remove outliers (values beyond 3 standard deviations)
    for col in columns_to_visualize:
        mean = df[col].mean()
        std = df[col].std()
        df = df[np.abs(df[col] - mean) <= (3 * std)]
    
    # Standardize the metrics
    scaler = RobustScaler()
    df_scaled = df.copy()
    df_scaled[columns_to_visualize] = scaler.fit_transform(df[columns_to_visualize])
    
    # Melt the dataframe for plotting
    df_melted = df_scaled.melt(
        id_vars=['is_slum', 'city'],
        value_vars=columns_to_visualize,
        var_name='Metric',
        value_name='Standardized Value'
    )
    
    # Convert boolean to string for better visualization
    df_melted['Area Type'] = df_melted['is_slum'].map({True: 'Slum', False: 'Non-Slum'})
    
    # Create figure
    plt.figure(figsize=(15, 8))
    
    # Create boxplot
    sns.boxplot(
        data=df_melted,
        x='Metric',
        y='Standardized Value',
        hue='Area Type',
        showfliers=False  # Hide outliers for cleaner visualization
    )
    
    # Customize plot
    plt.xticks(rotation=45, ha='right')
    plt.title('Standardized Distribution of Morphometric Metrics: Slum vs Non-Slum Areas', pad=20)
    plt.xlabel('')
    plt.ylabel('Standardized Value')
    
    # Add grid
    plt.grid(True, alpha=0.3, axis='y')
    
    # Adjust layout
    plt.tight_layout()
    
    # Calculate and print effect sizes
    print("\nEffect Sizes (Cohen's d) between Slum and Non-Slum areas:")
    for metric in columns_to_visualize:
        slum_values = df[df['is_slum']][metric]
        non_slum_values = df[~df['is_slum']][metric]
        
        # Calculate Cohen's d
        d = (slum_values.mean() - non_slum_values.mean()) / np.sqrt(
            ((len(slum_values) - 1) * slum_values.std()**2 + 
             (len(non_slum_values) - 1) * non_slum_values.std()**2) / 
            (len(slum_values) + len(non_slum_values) - 2)
        )
        
        print(f"{metric}: {d:.3f}")
    
    return plt.gcf()

# Define columns to visualize
columns_to_visualize = [
    'building_area',
    'perimeter',
    'elongation',
    'cell_elongation',
    'orientation',
    'cell_orientation',
    'fractal_dimension',
    'circular_compactness',
    'cell_circular_compactness',
    'convexity',
    'cell_convexity',
    'neighbor_distance',
    'coverage_ratio',
    'rectangularity',
    'cell_rectangularity',
    'building_adjacency',
    'alignment'
]

# Create and save the plot
fig = create_standardized_boxplots(combined_metrics, columns_to_visualize)
plt.savefig('plots/standardized_boxplots.png', dpi=300, bbox_inches='tight')
plt.close()

def create_city_wise_boxplots(combined_metrics, columns_to_visualize):
    """
    Create standardized boxplots for each city comparing slum vs non-slum areas.
    """
    # Filter columns and standardize metrics
    df = combined_metrics[columns_to_visualize + ['is_slum', 'city']].copy()
    
    # Remove outliers (values beyond 3 standard deviations)
    for col in columns_to_visualize:
        mean = df[col].mean()
        std = df[col].std()
        df = df[np.abs(df[col] - mean) <= (3 * std)]
    
    # Standardize the metrics
    scaler = RobustScaler()
    df[columns_to_visualize] = scaler.fit_transform(df[columns_to_visualize])
    
    # Melt the dataframe for plotting
    df_melted = df.melt(
        id_vars=['is_slum', 'city'],
        value_vars=columns_to_visualize,
        var_name='Metric',
        value_name='Standardized Value'
    )
    
    # Convert boolean to string for better visualization
    df_melted['Area Type'] = df_melted['is_slum'].map({True: 'Slum', False: 'Non-Slum'})
    
    # Get number of cities for subplot layout
    cities = df['city'].unique()
    n_cities = len(cities)
    
    # Calculate subplot layout
    n_cols = 2  # You can adjust this
    n_rows = int(np.ceil(n_cities / n_cols))
    
    # Create figure
    fig = plt.figure(figsize=(15 * n_cols, 8 * n_rows))
    
    # Create a plot for each city
    for idx, city in enumerate(cities, 1):
        plt.subplot(n_rows, n_cols, idx)
        
        # Filter data for current city
        city_data = df_melted[df_melted['city'] == city]
        
        # Create boxplot
        sns.boxplot(
            data=city_data,
            x='Metric',
            y='Standardized Value',
            hue='Area Type',
            showfliers=False  # Hide outliers for cleaner visualization
        )
        
        # Customize plot
        plt.xticks(rotation=45, ha='right')
        plt.title(f'{city}', pad=20)
        plt.xlabel('')
        plt.ylabel('Standardized Value')
        
        # Add grid
        plt.grid(True, alpha=0.3, axis='y')
        
        # Calculate effect sizes for this city
        print(f"\nEffect Sizes (Cohen's d) for {city}:")
        city_df = df[df['city'] == city]
        for metric in columns_to_visualize:
            slum_values = city_df[city_df['is_slum']][metric]
            non_slum_values = city_df[~city_df['is_slum']][metric]
            
            if len(slum_values) > 0 and len(non_slum_values) > 0:
                # Calculate Cohen's d
                d = (slum_values.mean() - non_slum_values.mean()) / np.sqrt(
                    ((len(slum_values) - 1) * slum_values.std()**2 + 
                     (len(non_slum_values) - 1) * non_slum_values.std()**2) / 
                    (len(slum_values) + len(non_slum_values) - 2)
                )
                print(f"{metric}: {d:.3f}")
    
    # Add overall title
    plt.suptitle('Standardized Distribution of Morphometric Metrics by City: Slum vs Non-Slum Areas', 
                 fontsize=16, y=1.02)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

# Create and save the plot
fig = create_city_wise_boxplots(combined_metrics, columns_to_visualize)
plt.savefig('plots/city_wise_standardized_boxplots.png', dpi=300, bbox_inches='tight')
plt.close()



#### MAPPING ####
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import box, Point
import os
import duckdb
from pyproj import Transformer

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

def create_square_bbox(lat, lon, size_meters):
    point = Point(lon, lat)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x, y = transformer.transform(lon, lat)
    half_size = size_meters / 2
    bbox = box(x - half_size, y - half_size, x + half_size, y + half_size)
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    minx, miny = transformer.transform(bbox.bounds[0], bbox.bounds[1])
    maxx, maxy = transformer.transform(bbox.bounds[2], bbox.bounds[3])
    return minx, miny, maxx, maxy

def plot_buildings_and_precarious_areas(ax, lat, lon, size_meters, city_name, grandparent_dir):
    xmin, ymin, xmax, ymax = create_square_bbox(lat, lon, size_meters)
    
    buildings_df = query_buildings_data(xmin, ymin, xmax, ymax)
    
    city_name_map = {
        'Sansalvador_Ps_': 'San Salvador, El Salvador',
        'SantoDomingo': 'Santo Domingo, Dominican Republic',
        'GuatemalaCity': 'Guatemala City, Guatemala',
        'Tegucigalpa': 'Tegucigalpa, Honduras',
        'SanJoseCRI': 'San Jose, Costa Rica',
        'Panama': 'Panama City, Panama',
        'BelizeCity': 'Belize City, Belize (excluded from the study)',
        'Managua': 'Managua, Nicaragua',
        'Belmopan_': 'Belmopan, Belize'
    }
    
    cities = {
        'Tegucigalpa': os.path.join(parent_dir, 'slums-model-unitac/data/SHP/Tegucigalpa_PS.shp'),
        'SantoDomingo': os.path.join(parent_dir, 'slums-model-unitac/data/0/SantoDomingo3857_buffered.geojson'),
        'GuatemalaCity': os.path.join(parent_dir, 'slums-model-unitac/data/SHP/Guatemala_PS.shp'),
        'Managua': os.path.join(parent_dir, 'slums-model-unitac/data/SHP/Managua_PS.shp'),
        'Panama': os.path.join(parent_dir, 'slums-model-unitac/data/SHP/Panama_PS.shp'),
        'BelizeCity': os.path.join(parent_dir, 'slums-model-unitac/data/SHP/BelizeCity_PS.shp'),
        'SanJoseCRI': os.path.join(parent_dir, 'slums-model-unitac/data/SHP/SanJose_PS.shp')
    }
        
    precarious_areas = gpd.read_file(cities[city_name])
    
    buildings_df = buildings_df.to_crs('EPSG:3857')
    precarious_areas = precarious_areas.to_crs('EPSG:3857')
    
    bbox = box(xmin, ymin, xmax, ymax)
    bbox_gdf = gpd.GeoDataFrame({'geometry': [bbox]}, crs='EPSG:4326').to_crs('EPSG:3857')
    
    precarious_areas = gpd.clip(precarious_areas, bbox_gdf)
    
    unified_precarious_area = precarious_areas.union_all()
    unified_precarious_gdf = gpd.GeoDataFrame({'geometry': [unified_precarious_area]}, crs='EPSG:3857')
    
    ax.set_facecolor('white')
    
    buildings_df.plot(ax=ax, edgecolor='none', facecolor='black', linewidth=0)
    
    # Plot precarious areas with faint fill and outline
    unified_precarious_gdf.plot(ax=ax, facecolor='red', edgecolor='red', alpha=0.05, linewidth=2)
    unified_precarious_gdf.boundary.plot(ax=ax, color='red', linewidth=2)
    
    ax.set_xlim(bbox_gdf.total_bounds[0], bbox_gdf.total_bounds[2])
    ax.set_ylim(bbox_gdf.total_bounds[1], bbox_gdf.total_bounds[3])
    
    ax.set_axis_off()
    
    cleaned_city_name = city_name_map.get(city_name, city_name)
    ax.set_title(f'{cleaned_city_name}', fontsize=14)

def plot_multiple_areas(coordinates_list, size_meters, grandparent_dir):
    n = len(coordinates_list)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    
    fig, axs = plt.subplots(rows, cols, figsize=(5*cols, 5*rows), squeeze=False)
    fig.suptitle('Building Footprints and Precarious Areas in Different Cities', fontsize=16)
    
    for i, (lat, lon, city_name) in enumerate(coordinates_list):
        row = i // cols
        col = i % cols
        ax = axs[row, col]
        plot_buildings_and_precarious_areas(ax, lat, lon, size_meters, city_name, grandparent_dir)
    
    # Hide any unused subplots
    for i in range(n, rows*cols):
        row = i // cols
        col = i % cols
        axs[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage
coordinates_list = [
    (14.643041208660227, -90.52696617369627, 'GuatemalaCity'),
    (14.09833557316007, -87.24716065494343, 'Tegucigalpa'),
    (18.506793321891678, -69.89322847545206, 'SantoDomingo'),
    (12.153548471297961, -86.25461143585959, 'Managua'),
    (8.925055642079421, -79.62752568376733, 'Panama'),
    (9.946122438272413, -84.08819439919527, 'SanJoseCRI'),
]

size_meters = 1500
from multiprocessing import freeze_support
freeze_support()
plot_multiple_areas(coordinates_list, size_meters, grandparent_dir)





### POPULATION VIS ###
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib import rcParams
import re

# Load the data
df = pd.read_csv('data/population_analysis.csv')

def clean_city_name(name):
    """Clean city names by adding spaces before capital letters."""
    return re.sub(r'(\w)([A-Z])', r'\1 \2', name)

# Filter for cities with valid data in both datasets
valid_data = df[df['urban_population'].notna() & (df['slum_population'] > 0)]

# Calculate non-slum population
valid_data['non_slum_population'] = valid_data['urban_population'] - valid_data['slum_population']
valid_data['clean_city'] = valid_data['city'].apply(clean_city_name)

# Create figure
fig, ax = plt.subplots(figsize=(18, 10))

# Set the width of each bar and positions of the bars
width = 0.35
cities = sorted(valid_data['city'].unique())
x = range(len(cities))

# Color schemes for each dataset
colors = {
    'HRSL': {'non_slum': '#1f77b4', 'slum': '#87CEEB'},  # Blue and light blue
    'GHS': {'non_slum': '#FFA500', 'slum': '#FFE4B5'}    # Orange and light orange
}

    
# Create bars for each dataset
for i, (dataset, color_dict) in enumerate(colors.items()):
    dataset_data = valid_data[valid_data['dataset'] == dataset].sort_values('city')
    
    # Create stacked bars
    non_slum_bars = ax.bar([xi + (i * width) for xi in x],
                          dataset_data['non_slum_population'],
                          width,
                          label=f'{dataset} Non-Slum',
                          color=color_dict['non_slum'])
    
    slum_bars = ax.bar([xi + (i * width) for xi in x],
                       dataset_data['slum_population'],
                       width,
                       bottom=dataset_data['non_slum_population'],
                       label=f'{dataset} Slum',
                       color=color_dict['slum'],
                       alpha=0.3)
    
    # Add percentage labels on top of bars
    for idx, (non_slum_bar, row) in enumerate(zip(non_slum_bars, dataset_data.itertuples())):
        total_height = row.urban_population
        percentage = (row.slum_population / row.urban_population) * 100
        ax.text(non_slum_bar.get_x() + non_slum_bar.get_width()/2,
                total_height,
                f'{percentage:.1f}%',
                ha='center', va='bottom',
                fontsize=10,
                fontweight='bold')

# Customize the plot
ax.set_title('Urban Population and Proportion in Precarious Areas by Dataset', 
            fontsize=22, pad=20)
ax.set_xlabel('City', fontsize=17)
ax.set_ylabel('Urban Population', fontsize=17)

# Set x-axis labels
ax.set_xticks([xi + width/2 for xi in x])
ax.set_xticklabels([clean_city_name(city) for city in cities], 
                   rotation=45, ha='right', fontsize=14)

# Format y-axis to millions
def millions_formatter(x, pos):
    return f'{x/1e6:.1f}M'
ax.yaxis.set_major_formatter(ticker.FuncFormatter(millions_formatter))

# Create custom legend
legend_elements = [
    plt.Rectangle((0,0), 1, 1, facecolor=colors['HRSL']['non_slum'], label='HRSL Non-precarious Population'),
    plt.Rectangle((0,0), 1, 1, facecolor=colors['HRSL']['slum'], alpha=0.3, label='HRSL Precarious Population'),
    plt.Rectangle((0,0), 1, 1, facecolor=colors['GHS']['non_slum'], label='GHS Non-precarious Population'),
    plt.Rectangle((0,0), 1, 1, facecolor=colors['GHS']['slum'], alpha=0.3, label='GHS Precarious Population'),
]
ax.legend(handles=legend_elements, fontsize=12, title='Population Type', 
         title_fontsize=14, loc='upper left')

# Add grid for easier comparison
ax.grid(axis='y', linestyle='--', alpha=0.3)

# Adjust layout and display
plt.tight_layout()

# Save the plot
plt.savefig('plots/population_proportion_comparison_stacked.png', dpi=300, bbox_inches='tight')