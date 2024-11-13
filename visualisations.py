import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Set the font family
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Garamond', 'Times New Roman', 'DejaVu Serif']

def clean_city_name(name):
    """Clean city names by adding spaces before capital letters."""
    return re.sub(r'(\w)([A-Z])', r'\1 \2', name)

def load_city_data(metrics_dir='metrics'):
    """Load all metrics data for each city."""
    city_data = {}
    
    # Walk through the metrics directory
    for city_name in os.listdir(metrics_dir):
        city_path = Path(metrics_dir) / city_name
        if city_path.is_dir():
            city_data[city_name] = {
                'building': pd.read_csv(city_path / 'building_metrics.csv'),
                'tessellation': pd.read_csv(city_path / 'tessellation_metrics.csv')
            }
    
    return city_data

def create_metric_plots(city_data, output_dir='plots'):
    """Create boxplots for each metric comparing slum and non-slum distributions."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set_palette("husl")
    
    # Define metrics to plot for each type
    building_metrics = [
        'building_area', 'perimeter', 'longest_axis', 'elongation', 
        'orientation', 'corners', 'fractal_dimension', 'squareness',
        'circular_compactness', 'convexity', 'rectangularity', 
        'building_adjacency', 'neighbor_distance'
    ]
    
    tessellation_metrics = [
        'cell_area', 'perimeter', 'circular_compactness', 'convexity',
        'orientation', 'elongation', 'rectangularity', 'car'
    ]
    
    # Process each city
    for city_name, data in city_data.items():
        print(f"Processing {city_name}...")
        
        # Create plots for building metrics
        fig, axes = plt.subplots(3, 5, figsize=(20, 12))
        fig.suptitle(f'Building Metrics Distribution - {city_name}', size=16)
        
        for idx, metric in enumerate(building_metrics):
            ax = axes[idx // 5, idx % 5] if idx < 15 else None
            if ax is not None and metric in data['building'].columns:
                sns.boxplot(x='is_slum', y=metric, data=data['building'], ax=ax)
                ax.set_title(metric)
                ax.set_xticklabels(['Non-Slum', 'Slum'])
        
        # Remove empty subplots
        for idx in range(len(building_metrics), 15):
            if idx < 15:
                axes[idx // 5, idx % 5].remove()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{city_name}_building_metrics.png'))
        plt.close()
        
        # Create plots for tessellation metrics
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'Tessellation Metrics Distribution - {city_name}', size=16)
        
        for idx, metric in enumerate(tessellation_metrics):
            ax = axes[idx // 4, idx % 4] if idx < 8 else None
            if ax is not None and metric in data['tessellation'].columns:
                sns.boxplot(x='is_slum', y=metric, data=data['tessellation'], ax=ax)
                ax.set_title(metric)
                ax.set_xticklabels(['Non-Slum', 'Slum'])
        
        # Remove empty subplots
        for idx in range(len(tessellation_metrics), 8):
            if idx < 8:
                axes[idx // 4, idx % 4].remove()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{city_name}_tessellation_metrics.png'))
        plt.close()

def create_combined_plots(city_data, output_dir='plots'):
    """Create combined plots comparing all cities."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine all data
    building_dfs = []
    tessellation_dfs = []
    
    for city_name, data in city_data.items():
        # Add city name to each dataframe
        building_df = data['building'].copy()
        building_df['city'] = city_name
        building_dfs.append(building_df)
        
        tessellation_df = data['tessellation'].copy()
        tessellation_df['city'] = city_name
        tessellation_dfs.append(tessellation_df)
    
    combined_building = pd.concat(building_dfs, ignore_index=True)
    combined_tessellation = pd.concat(tessellation_dfs, ignore_index=True)
    
    # Create violin plots for selected metrics
    key_metrics = {
        'building': ['building_area', 'circular_compactness', 'neighbor_distance'],
        'tessellation': ['cell_area', 'circular_compactness', 'car']
    }
    
    for data_type, metrics in key_metrics.items():
        data = combined_building if data_type == 'building' else combined_tessellation
        
        plt.figure(figsize=(15, 10))
        for idx, metric in enumerate(metrics):
            plt.subplot(1, 3, idx + 1)
            sns.violinplot(x='city', y=metric, hue='is_slum', data=data, split=True)
            plt.xticks(rotation=45)
            plt.title(metric)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'combined_{data_type}_metrics.png'))
        plt.close()

# Load all city data
city_data = load_city_data()

# Create individual city plots
create_metric_plots(city_data)

# Create combined comparison plots
create_combined_plots(city_data)

def load_and_process_city_data(metrics_dir='metrics'):
    """Load and process metrics data with standardized city names."""
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
    
    # Load data
    city_data = {}
    for city_name in os.listdir(metrics_dir):
        city_path = Path(metrics_dir) / city_name
        if city_path.is_dir():
            # Load and combine metrics
            building_df = pd.read_csv(city_path / 'building_metrics.csv')
            tessellation_df = pd.read_csv(city_path / 'tessellation_metrics.csv')
            
            # Add prefix to avoid column name conflicts
            building_df.columns = ['building_' + col if col not in ['is_slum'] else col for col in building_df.columns]
            tessellation_df.columns = ['tessellation_' + col if col not in ['is_slum'] else col for col in tessellation_df.columns]
            
            # Combine metrics
            combined_df = pd.concat([building_df, tessellation_df], axis=1)
            combined_df['city'] = city_name_map.get(city_name, city_name)
            city_data[city_name] = combined_df
    
    return pd.concat(city_data.values(), ignore_index=True)

def create_consolidated_plots(data, output_dir='plots'):
    """Create consolidated plots for each city comparing slum and non-slum distributions."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Define key metrics to visualize
    metrics = {
        'Building Metrics': {
            'building_area': 'Building Area',
            'building_circular_compactness': 'Building Compactness',
            'building_neighbor_distance': 'Neighbor Distance',
            'building_adjacency': 'Building Adjacency'
        },
        'Tessellation Metrics': {
            'tessellation_cell_area': 'Cell Area',
            'tessellation_car': 'Coverage Area Ratio',
            'tessellation_circular_compactness': 'Cell Compactness',
            'tessellation_rectangularity': 'Cell Rectangularity'
        }
    }
    
    # Set style
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Garamond'] + plt.rcParams['font.serif']
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.color'] = '#E6E6E6'
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    
    # Process each city
    for city_name in data['city'].unique():
        city_data = data[data['city'] == city_name]
        
        # Create separate plots for building and tessellation metrics
        for metric_type, metric_dict in metrics.items():
            fig, axes = plt.subplots(figsize=(15, 10))
            
            # Prepare data for plotting
            plot_data = []
            for metric, display_name in metric_dict.items():
                if metric in city_data.columns:
                    # Standardize the metric
                    scaler = MinMaxScaler()
                    standardized_values = scaler.fit_transform(city_data[metric].values.reshape(-1, 1))
                    
                    # Create DataFrame for this metric
                    metric_df = pd.DataFrame({
                        'Metric': display_name,
                        'Value': standardized_values.flatten(),
                        'Type': ['Slum' if is_slum else 'Non-Slum' for is_slum in city_data['is_slum']]
                    })
                    plot_data.append(metric_df)
            
            # Combine all metrics
            plot_df = pd.concat(plot_data, ignore_index=True)
            
            # Create violin plot
            sns.violinplot(x='Metric', y='Value', hue='Type', data=plot_df, 
                         split=True, inner='box', ax=axes)
            
            # Customize plot
            plt.title(f'{metric_type} Distribution - {city_name}', fontsize=16, pad=20)
            plt.xlabel('Metric', fontsize=12, labelpad=10)
            plt.ylabel('Standardized Value', fontsize=12, labelpad=10)
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Area Type', bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{city_name}_{metric_type.lower().replace(" ", "_")}.png'),
                       bbox_inches='tight', dpi=300)
            plt.close()

data = load_and_process_city_data()
create_consolidated_plots(data)


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


# Set the font family
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Garamond', 'Times New Roman', 'DejaVu Serif']

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