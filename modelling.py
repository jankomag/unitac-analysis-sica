import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix, 
                           precision_recall_curve, average_precision_score,
                           roc_curve, roc_auc_score)
import glob

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)

def load_combined_metrics(metrics_dir='metrics/bycity/'):
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

# Calculate missing values
missing_values = combined_metrics.isnull().sum()
missing_percentages = (missing_values / len(combined_metrics)) * 100

# Create a DataFrame for the missing values
missing_df = pd.DataFrame({
    'Column': missing_values.index,
    'Missing Values': missing_values.values,
    'Percentage': missing_percentages.values
})

# Sort by number of missing values
missing_df = missing_df.sort_values('Missing Values', ascending=False)

# Keep only columns with missing values
missing_df = missing_df[missing_df['Missing Values'] > 0]


# remove NAs
original_count = len(combined_metrics)
clean_metrics = combined_metrics.dropna()
cleaned_count = len(clean_metrics)
removed_count = original_count - cleaned_count
print(f"\nData Row Comparison:")
print(f"Original rows: {original_count:,}")
print(f"Rows removed:  {removed_count:,} ({(removed_count/original_count)*100:.1f}%)")
print(f"Rows remaining: {cleaned_count:,} ({(cleaned_count/original_count)*100:.1f}%)")


if len(missing_df) > 0:
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Bar plot of missing values
    sns.barplot(data=missing_df, x='Column', y='Missing Values', ax=ax1, color='royalblue')
    ax1.set_title('Number of Missing Values by Column')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Heatmap of missing values
    plt.subplot(212)
    sns.heatmap(combined_metrics[missing_df['Column']].isnull(), 
                cmap='YlOrRd', 
                cbar_kws={'label': 'Missing'},
                yticklabels=False)
    ax2.set_title('Missing Values Pattern')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('plots/missing_values.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    print("\nMissing Values Summary:")
    print(missing_df.to_string(index=False))
    
    # Print missing values by city
    print("\nMissing Values by City:")
    city_missing = combined_metrics.groupby('city').apply(lambda x: x.isnull().sum())
    print(city_missing[city_missing > 0].to_string())
else:
    print("\nNo missing values found in the dataset!")

# Print total number of rows for context
print(f"\nTotal number of rows in dataset: {len(combined_metrics)}")

# Use the features we identified from your code
features = [
    'building_area',
    'perimeter',
    'elongation',
    'fractal_dimension',
    'circular_compactness',
    'rectangularity',
]

# Print class distribution
print("Class distribution in full dataset:")
print(combined_metrics['is_slum'].value_counts(normalize=True))

# Create stratification column combining city and slum status
combined_metrics['strat'] = combined_metrics['city'] + '_' + combined_metrics['is_slum'].astype(str)

# Print class distribution
print("Class distribution in full dataset:")
print(combined_metrics['is_slum'].value_counts(normalize=True))

# Create stratification column combining city and slum status
combined_metrics['strat'] = combined_metrics['city'] + '_' + combined_metrics['is_slum'].astype(str)

# Prepare the data
X = combined_metrics[features]
y = combined_metrics['is_slum']
strat = combined_metrics['strat']

# Initialize the cross-validation splitter
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize lists to store results
fold_metrics = []
fold_feature_importance = []
city_performance = {city: [] for city in combined_metrics['city'].unique()}

# Initialize arrays for aggregated predictions
all_probs = []
all_true = []

# Perform cross-validation
for fold, (train_idx, val_idx) in enumerate(cv.split(X, strat), 1):
    # Split data
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_val_scaled)
    y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
    
    # Store probabilities and true values for later
    all_probs.extend(y_pred_proba)
    all_true.extend(y_val)
    
    # Get classification report
    report = classification_report(y_val, y_pred, output_dict=True)
    
    # Calculate metrics for this fold
    fold_metrics.append({
        'fold': fold,
        'roc_auc': roc_auc_score(y_val, y_pred_proba),
        'avg_precision': average_precision_score(y_val, y_pred_proba),
        'precision_slum': report['True']['precision'],
        'recall_slum': report['True']['recall'],
        'f1_slum': report['True']['f1-score'],
        'support_slum': report['True']['support']
    })
    
    # Store feature importance
    fold_feature_importance.append(model.coef_[0])
    
    # Calculate city-wise performance
    print(f"\nFold {fold} Results by City:")
    for city in combined_metrics['city'].unique():
        city_mask = combined_metrics.iloc[val_idx]['city'] == city
        if sum(city_mask) > 0:
            city_y_true = y_val[city_mask]
            city_y_pred = y_pred[city_mask]
            city_y_proba = y_pred_proba[city_mask]
            
            if len(np.unique(city_y_true)) > 1:  # Check if both classes are present
                city_report = classification_report(city_y_true, city_y_pred, output_dict=True)
                city_metrics = {
                    'roc_auc': roc_auc_score(city_y_true, city_y_proba),
                    'precision': city_report['True']['precision'],
                    'recall': city_report['True']['recall'],
                    'f1': city_report['True']['f1-score'],
                    'support': city_report['True']['support']
                }
                city_performance[city].append(city_metrics)
                print(f"\n{city}:")
                print(f"ROC-AUC: {city_metrics['roc_auc']:.3f}")
                print(f"Precision (Slum): {city_metrics['precision']:.3f}")
                print(f"Recall (Slum): {city_metrics['recall']:.3f}")
                print(f"F1 (Slum): {city_metrics['f1']:.3f}")

    # Print fold results
    print(f"\nFold {fold} Overall Results:")
    print(classification_report(y_val, y_pred))
    
    # Plot confusion matrix with percentages
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_val, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues')
    plt.title(f'Normalized Confusion Matrix - Fold {fold}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'plots/confusion_matrix_fold_{fold}.png', dpi=300, bbox_inches='tight')
    plt.close()

# Calculate and plot overall ROC curve
plt.figure(figsize=(10, 8))
fpr, tpr, _ = roc_curve(all_true, all_probs)
roc_auc = roc_auc_score(all_true, all_probs)
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig('plots/roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot Precision-Recall curve
plt.figure(figsize=(10, 8))
precision, recall, _ = precision_recall_curve(all_true, all_probs)
avg_precision = average_precision_score(all_true, all_probs)
plt.plot(recall, precision, label=f'PR curve (AP = {avg_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower right")
plt.savefig('plots/precision_recall_curve.png', dpi=300, bbox_inches='tight')
plt.close()

# Print summary of results
metrics_df = pd.DataFrame(fold_metrics)
print("\nOverall Cross-Validation Results:")
print("\nMean metrics across folds:")
print(metrics_df.mean())
print("\nStandard deviation of metrics across folds:")
print(metrics_df.std())

# Print city-wise summary
print("\nCity-wise Performance Summary:")
for city, metrics_list in city_performance.items():
    if metrics_list:
        metrics_array = pd.DataFrame(metrics_list)
        print(f"\n{city}:")
        print("Mean metrics:")
        print(metrics_array.mean())
        print("\nStd metrics:")
        print(metrics_array.std())