import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)


def clean_city_name(name):
    """Clean city names by adding spaces before capital letters."""
    return re.sub(r'(\w)([A-Z])', r'\1 \2', name)

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

def create_comparative_boxplots(cities_input=None):
    """
    Create boxplots comparing metrics between slum and non-slum areas across all cities.
    Merges building and tessellation metrics based on uID.
    """
    # Metrics to visualize (we'll update this list after merging)
    building_metrics = [
        'building_area', 'perimeter', 'longest_axis', 'elongation', 'orientation',
        'corners', 'fractal_dimension', 'squareness', 'circular_compactness',
        'convexity', 'rectangularity', 'building_adjacency', 'neighbor_distance'
    ]
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Initialize lists to store data
    all_data = []
    
    # Determine which cities to process
    if cities_input is None:
        cities = [d for d in os.listdir('metrics') if os.path.isdir(os.path.join('metrics', d))]
    elif isinstance(cities_input, dict):
        cities = list(cities_input.keys())
    elif isinstance(cities_input, list):
        cities = cities_input
    else:
        raise ValueError("cities_input must be None, a dictionary, or a list")
    
    # Load and merge data for each city
    for city in cities:
        try:
            # Load both datasets
            buildings_df = pd.read_csv(f'metrics/{city}/building_metrics.csv')
            tessellation_df = pd.read_csv(f'metrics/{city}/tessellation_metrics.csv')
            
            # Add prefix to tessellation columns (except uID and is_slum)
            tessellation_cols = tessellation_df.columns.difference(['uID', 'is_slum'])
            tessellation_df = tessellation_df.rename(
                columns={col: f'cell_{col}' for col in tessellation_cols}
            )
            
            # Merge datasets on uID
            merged_df = pd.merge(buildings_df, tessellation_df, on='uID', 
                               suffixes=('', '_y'))
            
            # If is_slum appears in both, keep one version
            if 'is_slum_y' in merged_df.columns:
                merged_df = merged_df.drop(columns='is_slum_y')
            
            merged_df['city'] = city
            all_data.append(merged_df)
            print(f"Loaded and merged data for {city}")
            
            # Update metrics list for the first city to include new cell_ metrics
            if len(all_data) == 1:
                # Add cell metrics to visualization
                cell_metrics = [col for col in merged_df.columns if col.startswith('cell_')]
                building_metrics.extend(cell_metrics)
                
        except Exception as e:
            print(f"Could not load data for {city}: {str(e)}")
            continue
    
    if not all_data:
        raise ValueError("No data was loaded for any city")
        
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Create figure with subplots
    n_metrics = len(building_metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(20, 5*n_rows))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # Create boxplots for each metric
    for idx, metric in enumerate(building_metrics, 1):
        plt.subplot(n_rows, n_cols, idx)
        
        # Handle potential outliers by limiting to 95th percentile
        upper_limit = combined_df[metric].quantile(0.95)
        plot_data = combined_df[combined_df[metric] <= upper_limit].copy()
        
        # Create boxplot
        sns.boxplot(data=plot_data, x='city', y=metric, hue='is_slum', 
                   showfliers=False)
        
        # Customize plot
        plt.xticks(rotation=45, ha='right')
        plt.title(f'{metric.replace("_", " ").title()}')
        plt.xlabel('')
        plt.ylabel(metric)
        
        # Update legend
        if idx == 1:
            plt.legend(title='Slum Area', labels=['Non-Slum', 'Slum'])
        else:
            plt.legend([],[], frameon=False)
    
    plt.suptitle('Comparison of Urban Morphology Metrics in Slum vs Non-Slum Areas', 
                 fontsize=16, y=1.02)
    
    plt.tight_layout()
    plt.savefig('plots/all_cities_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("All cities comparison saved as 'plots/all_cities_comparison.png'")

def create_individual_city_plots(cities_input=None):
    """
    Create separate plots for each city comparing all metrics between slum and non-slum areas.
    """
    building_metrics = [
        'building_area', 'perimeter', 'longest_axis', 'elongation', 'orientation',
        'corners', 'fractal_dimension', 'squareness', 'circular_compactness',
        'convexity', 'rectangularity', 'building_adjacency', 'neighbor_distance'
    ]
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Determine which cities to process
    if cities_input is None:
        cities = [d for d in os.listdir('metrics') if os.path.isdir(os.path.join('metrics', d))]
    elif isinstance(cities_input, dict):
        cities = list(cities_input.keys())
    elif isinstance(cities_input, list):
        cities = cities_input
    else:
        raise ValueError("cities_input must be None, a dictionary, or a list")
    
    # Process each city
    for city in cities:
        try:
            # Load city data
            df = pd.read_csv(f'metrics/{city}/building_metrics.csv')
            print(f"Processing {city}")
            
            # Create figure
            fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(15, 25))
            plt.subplots_adjust(hspace=0.4)
            
            # Flatten axes for easier iteration
            axes_flat = axes.flatten()
            
            # Create violin plots for each metric
            for idx, metric in enumerate(building_metrics):
                if idx >= len(axes_flat):
                    break
                    
                ax = axes_flat[idx]
                
                # Handle potential outliers
                upper_limit = df[metric].quantile(0.95)
                plot_data = df[df[metric] <= upper_limit].copy()
                
                # Create violin plot with overlaid boxplot
                sns.violinplot(data=plot_data, x='is_slum', y=metric, ax=ax,
                             inner='box', cut=0)
                
                # Customize plot
                ax.set_title(metric.replace('_', ' ').title())
                ax.set_xlabel('')
                ax.set_xticklabels(['Non-Slum', 'Slum'])
                
                # Add median values as text
                medians = plot_data.groupby('is_slum')[metric].median()
                for i, median in enumerate(medians):
                    ax.text(i, ax.get_ylim()[0], f'Med: {median:.2f}', 
                           horizontalalignment='center', verticalalignment='top')
            
            # Remove empty subplots if any
            for idx in range(len(building_metrics), len(axes_flat)):
                fig.delaxes(axes_flat[idx])
            
            plt.suptitle(f'Morphological Metrics Comparison for {city}', 
                        fontsize=16, y=1.02)
            
            plt.tight_layout()
            plt.savefig(f'plots/{city}_metrics.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved plot for {city}")
            
        except Exception as e:
            print(f"Error processing {city}: {str(e)}")
            continue

# Run both visualizations
create_comparative_boxplots(list(cities.keys()))
create_individual_city_plots(list(cities.keys()))

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
import numpy as np
import os

def create_combined_scaled_plots(cities_input=None):
    """
    Create one long plot per city with all metrics (building and cell) scaled to the same range,
    using boxplots with improved visibility
    """
    # Define metrics to include
    building_metrics = [
        'building_area', 'perimeter', 'longest_axis', 'elongation', 'orientation',
        'corners', 'fractal_dimension', 'squareness', 'circular_compactness',
        'convexity', 'building_adjacency', 'neighbor_distance' #rectangularity
    ]
    
    cell_metrics = [
        'cell_area', 'perimeter', 'circular_compactness', 'convexity',
        'orientation', 'elongation', 'rectangularity', 'car'
    ]
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
        
    # Determine which cities to process
    if cities_input is None:
        cities = [d for d in os.listdir('metrics') if os.path.isdir(os.path.join('metrics', d))]
    elif isinstance(cities_input, dict):
        cities = list(cities_input.keys())
    elif isinstance(cities_input, list):
        cities = cities_input
    else:
        raise ValueError("cities_input must be None, a dictionary, or a list")
    
    # Process each city
    for city in cities:
        try:
            # Load building and cell data
            buildings_df = pd.read_csv(f'metrics/{city}/building_metrics.csv')
            cells_df = pd.read_csv(f'metrics/{city}/tessellation_metrics.csv')
            
            print(f"Processing {city}")
            
            # Prepare building metrics
            building_data = []
            scaler = RobustScaler()  # Use RobustScaler instead of StandardScaler
            
            for metric in building_metrics:
                if metric in buildings_df.columns:
                    # Remove extreme outliers before scaling
                    metric_data = buildings_df[metric]
                    Q1 = metric_data.quantile(0.05)
                    Q3 = metric_data.quantile(0.95)
                    IQR = Q3 - Q1
                    
                    # Filter out extreme outliers
                    valid_mask = (metric_data >= Q1 - 3 * IQR) & (metric_data <= Q3 + 3 * IQR)
                    filtered_data = metric_data[valid_mask]
                    filtered_is_slum = buildings_df['is_slum'][valid_mask]
                    
                    # Scale the filtered data
                    scaled_values = scaler.fit_transform(filtered_data.values.reshape(-1, 1))
                    
                    # Create a row for each value
                    for val, is_slum in zip(scaled_values, filtered_is_slum, strict=False):
                        building_data.append({
                            'Metric': f'Building_{metric}',
                            'Value': val[0],
                            'Type': 'Building',
                            'Is_Slum': bool(is_slum)
                        })
            
            # Prepare cell metrics
            cell_data = []
            for metric in cell_metrics:
                if metric in cells_df.columns:
                    # Remove extreme outliers before scaling
                    metric_data = cells_df[metric]
                    Q1 = metric_data.quantile(0.2)
                    Q3 = metric_data.quantile(0.8)
                    IQR = Q3 - Q1
                    
                    # Filter out extreme outliers
                    valid_mask = (metric_data >= Q1 - 3 * IQR) & (metric_data <= Q3 + 3 * IQR)
                    filtered_data = metric_data[valid_mask]
                    filtered_is_slum = cells_df['is_slum'][valid_mask]
                    
                    # Scale the filtered data
                    scaled_values = scaler.fit_transform(filtered_data.values.reshape(-1, 1))
                    
                    # Create a row for each value
                    for val, is_slum in zip(scaled_values, filtered_is_slum, strict=False):
                        cell_data.append({
                            'Metric': f'Cell_{metric}',
                            'Value': val[0],
                            'Type': 'Cell',
                            'Is_Slum': bool(is_slum)
                        })
            
            # Combine all data
            all_data = pd.DataFrame(building_data + cell_data)
            
            # Create the plot
            plt.figure(figsize=(15, 20))
            
            # Create box plot with custom style
            sns.boxplot(data=all_data, x='Value', y='Metric', hue='Is_Slum',
                       orient='h', 
                       showfliers=True,  # Show moderate outliers
                       fliersize=2,  # Make outlier points smaller
                       linewidth=2,  # Make box lines thicker
                       palette=['lightblue', 'lightcoral'])  # Use distinct colors
            
            # Customize plot
            plt.title(f'Morphological Metrics Comparison for {city}', 
                     pad=20, fontsize=16, fontweight='bold')
            plt.xlabel('Standardized Value', fontsize=12)
            plt.ylabel('Metric', fontsize=12)
            
            # Improve metric labels
            current_labels = plt.gca().get_yticklabels()
            new_labels = [label.get_text().replace('Building_', '').replace('Cell_', '') 
                         for label in current_labels]
            plt.gca().set_yticklabels(new_labels, fontsize=10)
            
            # Add separator lines between building and cell metrics
            last_building_metric = None
            for i, metric in enumerate(plt.gca().get_yticklabels()):
                if metric.get_text().startswith('Cell') and last_building_metric is None:
                    last_building_metric = i
                    plt.axhline(y=i - 0.5, color='gray', linestyle='--', alpha=0.5)
            
            # Improve legend
            plt.legend(title='Area Type', labels=['Non-Slum', 'Slum'], 
                      title_fontsize=12, fontsize=10)
            
            # Add vertical reference line at 0
            plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            
            # Add grid but make it lighter
            plt.grid(True, alpha=0.2)
            
            # Set background color
            plt.gca().set_facecolor('white')
            
            # Add subtle box around plot
            plt.box(True)
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(f'plots/{city}_combined_metrics.png', 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"Saved combined plot for {city}")
            
        except Exception as e:
            print(f"Error processing {city}: {str(e)}")
            continue

# Run the visualization
create_combined_scaled_plots(list(cities.keys()))



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
    'HRSL': {'non_slum': '#1f77b4', 'slum': '#1f77b4'},
    'GHS': {'non_slum': '#2ca02c', 'slum': '#2ca02c'}
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
ax.set_title('Urban Population and Proportion in Slum Areas by Dataset', 
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
    plt.Rectangle((0,0), 1, 1, facecolor=colors['HRSL']['non_slum'], label='HRSL Total Population'),
    plt.Rectangle((0,0), 1, 1, facecolor=colors['HRSL']['slum'], alpha=0.3, label='HRSL Slum Population'),
    plt.Rectangle((0,0), 1, 1, facecolor=colors['GHS']['non_slum'], label='GHS Total Population'),
    plt.Rectangle((0,0), 1, 1, facecolor=colors['GHS']['slum'], alpha=0.3, label='GHS Slum Population'),
]
ax.legend(handles=legend_elements, fontsize=12, title='Population Type', 
         title_fontsize=14, loc='upper right')

# Add grid for easier comparison
ax.grid(axis='y', linestyle='--', alpha=0.3)

# Adjust layout and display
plt.tight_layout()

# Save the plot
plt.savefig('plots/population_proportion_comparison_stacked.png', dpi=300, bbox_inches='tight')