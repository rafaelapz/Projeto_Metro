

import pandas as pd
import numpy as np
import json
import sys
import os
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error
from scipy import stats
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
warnings.filterwarnings('ignore')


# Feature Selection Parameters
CORRELATION_THRESHOLD = 0.7     # Remove features with correlation > this value
MAX_FEATURES = 10            # Maximum features to select for GraphSAGE
MIN_FEATURES = 2        # Minimum features to keep

# Statistical Validation Parameters
SKEWNESS_THRESHOLD = 2.0        # Features with |skew| > this may need transformation
OUTLIER_THRESHOLD = 0.05         # Max proportion of outliers allowed
MIN_UNIQUE_VALUES = 20         # Minimum unique values for continuous treatment

# Priority Features - Always keep these when possible
PRIORITY_KEYWORDS = ['empregos', 'residencia']

# Cross-Validation Parameters
N_FOLDS = 5              # Number of folds for cross-validation
RANDOM_STATE = 42                # Random seed for reproducibility

# Random Forest Parameters (for feature importance)
RF_PARAMS_ZONE = {
    'n_estimators': 100,
    'max_depth': 6,              # Can be deeper with 527 samples
    'min_samples_split': 10,     # Less strict with more samples
    'min_samples_leaf': 5,       # Less strict with more samples
    'max_features': 0.3,         # Use only 30% of features at each split
    'random_state': RANDOM_STATE,
    'oob_score': True
}

RF_PARAMS_STATION = {
    'n_estimators': 100,
    'max_depth': 4,              # Smaller depth for 168 samples
    'min_samples_split': 20,     # Higher for regularization
    'min_samples_leaf': 10,      # Higher for regularization
    'max_features': 0.3,
    'random_state': RANDOM_STATE,
    'oob_score': True
}

# Feature Importance Thresholds
MIN_IMPORTANCE = 0.01            # Minimum importance to consider a feature
IMPORTANCE_STABILITY_WEIGHT = 0.7 # Weight for mean importance vs stability (0-1)



sys.path.append('/Users/ellazyngier/Documents/github/tccII/scripts/dados')
from dados.linhas import current_and_express_lines
#ZONES_PATH = '/Users/ellazyngier/Documents/github/tccII/scripts/dados/teste.xlsx'
ZONES_PATH = '/Users/ellazyngier/Documents/github/tccII/scripts/dados/dados_zonas.xlsx'
#ZONES_PATH = '/Users/ellazyngier/Documents/github/tccII/scripts/dados/dados_zonas_so_dois.xlsx'
METRO_PATH = '/Users/ellazyngier/Documents/github/tccII/scripts/dados/metro_e_trem_para_treinar.xlsx'
STATIONS_PATH = '/Users/ellazyngier/Documents/github/tccII/scripts/dados/station_zones_882.json'
OUTPUT_DIR = '/Users/ellazyngier/Documents/github/tccII/scripts/resultados'
DADOS_DIR = '/Users/ellazyngier/Documents/github/tccII/scripts/dados'

os.makedirs(OUTPUT_DIR, exist_ok=True)


def print_section(title, char="="):
    """Print formatted section header."""
    print("\n" + char*70)
    print(title)
    print(char*70)

def load_data():
    """Load zone data, metro trips, and station coverage."""
    print_section("LOADING DATA")
    
    zones_df = pd.read_excel(ZONES_PATH)
    metro_df = pd.read_excel(METRO_PATH)
    
    # Convert to numeric
    for col in zones_df.columns:
        if col != 'zona':
            zones_df[col] = pd.to_numeric(zones_df[col], errors='coerce')
    
    for col in metro_df.columns:
        if col != 'zona':
            metro_df[col] = pd.to_numeric(metro_df[col], errors='coerce')
    
    zones_df = zones_df.fillna(zones_df.median())
    
    # Merge with target
    target_col = metro_df.columns[3]
    merged = pd.merge(zones_df, metro_df[['zona', target_col]], on='zona', how='inner')
    merged = merged.rename(columns={target_col: 'metro_trips'})
    
    # Load station coverage
    with open(STATIONS_PATH, 'r') as f:
        station_coverage = json.load(f)
    
    print(f"✅ Loaded {len(merged)} zones with {len(zones_df.columns)-1} features")
    print(f"✅ Station coverage data for {len(station_coverage)} stations")
    print(f"✅ Target variable: {target_col}")
    
    return merged, station_coverage

def get_active_stations():
    """Get list of all active stations from current lines."""
    active_stations = set()
    for line in current_and_express_lines:
        active_stations.update(line)
    return sorted(list(active_stations))

def check_pearson_appropriateness(X, create_plots=False):
    """
    Check if features are appropriate for Pearson correlation.
    
    Based on Schober et al. (2018): "Correlation Coefficients: 
    Appropriate Use and Interpretation"
    
    Returns dict with recommendations for each feature.
    """
    print_section("CHECKING PEARSON CORRELATION APPROPRIATENESS")
    
    recommendations = {}
    problematic_features = []
    
    print("\n📊 Feature Analysis:")
    print("-" * 70)
    print(f"{'Feature':<30} {'Skew':>8} {'Outliers':>10} {'Unique':>8} {'Status':<10}")
    print("-" * 70)
    
    for col in X.columns:
        # 1. Check skewness
        skewness = X[col].skew()
        
        # 2. Check for outliers using IQR method
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((X[col] < Q1 - 3*IQR) | (X[col] > Q3 + 3*IQR)).sum()
        outlier_pct = outliers / len(X)
        
        # 3. Check number of unique values (discreteness)
        unique_values = X[col].nunique()
        
        # 4. Normality test (Shapiro-Wilk for small samples)
        if len(X) < 50:
            stat, p_value = stats.shapiro(X[col])
            is_normal = p_value > 0.05
        else:
            # For larger samples, use D'Agostino-Pearson test
            stat, p_value = stats.normaltest(X[col])
            is_normal = p_value > 0.05
        
        # Determine if Pearson is appropriate
        issues = []
        use_pearson = True
        
        if abs(skewness) > SKEWNESS_THRESHOLD:
            issues.append("skewed")
            use_pearson = False
        
        if outlier_pct > OUTLIER_THRESHOLD:
            issues.append("outliers")
            use_pearson = False
        
        if unique_values < MIN_UNIQUE_VALUES:
            issues.append("discrete")
            use_pearson = False
        
        # Store recommendation
        recommendations[col] = {
            'use_pearson': use_pearson,
            'skewness': skewness,
            'outlier_pct': outlier_pct,
            'unique_values': unique_values,
            'is_normal': is_normal,
            'issues': issues
        }
        
        if not use_pearson:
            problematic_features.append(col)
        
        # Print summary (show only first 30 features for brevity)
        if X.columns.get_loc(col) < 30 or use_pearson == False:
            status = "✅ OK" if use_pearson else f"⚠️ {','.join(issues)}"
            print(f"{col[:30]:<30} {skewness:>8.2f} {outlier_pct:>9.1%} {unique_values:>8} {status:<10}")
    
    if len(X.columns) > 30:
        print("... [showing first 30 and problematic features only]")
    
    print("-" * 70)
    
    # Summary statistics
    pearson_appropriate = sum(1 for r in recommendations.values() if r['use_pearson'])
    total_features = len(recommendations)
    
    print(f"\n📈 Summary:")
    print(f"   Features appropriate for Pearson: {pearson_appropriate}/{total_features} ({pearson_appropriate/total_features*100:.1f}%)")
    
    if problematic_features:
        print(f"\n⚠️  Problematic features ({len(problematic_features)}):")
        for feat in problematic_features[:10]:  # Show first 10
            issues = recommendations[feat]['issues']
            print(f"   • {feat}: {', '.join(issues)}")
        
        if len(problematic_features) > 10:
            print(f"   ... and {len(problematic_features) - 10} more")
        
        print(f"\n💡 Recommendations:")
        print(f"   1. Will use Spearman correlation for these features")
        print(f"   2. Pearson features compared only with Pearson features")
        print(f"   3. Spearman features compared only with Spearman features")
    
    # Create diagnostic plots if requested
    if create_plots and problematic_features:
        create_diagnostic_plots(X, problematic_features[:6])
    
    return recommendations

def create_diagnostic_plots(X, features_to_plot):
    """Create diagnostic plots for problematic features."""
    n_features = min(len(features_to_plot), 6)
    features_to_plot = features_to_plot[:n_features]
    
    fig, axes = plt.subplots(n_features, 3, figsize=(12, 4*n_features))
    
    if n_features == 1:
        axes = axes.reshape(1, -1)
    
    for i, col in enumerate(features_to_plot):
        # Histogram
        axes[i, 0].hist(X[col], bins=20, edgecolor='black', alpha=0.7)
        axes[i, 0].set_title(f'{col[:20]}\nDistribution')
        axes[i, 0].set_xlabel('Value')
        axes[i, 0].set_ylabel('Frequency')
        
        # Q-Q plot
        stats.probplot(X[col], dist="norm", plot=axes[i, 1])
        axes[i, 1].set_title('Q-Q Plot')
        
        # Box plot
        axes[i, 2].boxplot(X[col])
        axes[i, 2].set_title('Box Plot')
        axes[i, 2].set_ylabel('Value')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'diagnostic_plots.png'))
    plt.close()
    print(f"   Diagnostic plots saved to {OUTPUT_DIR}/diagnostic_plots.png")

def decide_which_to_drop(X, feat1, feat2):
    """
    Decide which feature to drop, prioritizing employment and residence features.
    """
    # Check if either feature contains priority keywords
    feat1_priority = any(kw in feat1.lower() for kw in PRIORITY_KEYWORDS)
    feat2_priority = any(kw in feat2.lower() for kw in PRIORITY_KEYWORDS)
    
    # If one has priority and other doesn't, keep the priority one
    if feat1_priority and not feat2_priority:
        return feat2  # Drop feat2, keep feat1
    elif feat2_priority and not feat1_priority:
        return feat1  # Drop feat1, keep feat2
    
    # If both or neither have priority, use variance as before
    if X[feat1].var() < X[feat2].var():
        return feat1
    else:
        return feat2

def remove_correlated_features_separate_groups(X, threshold=CORRELATION_THRESHOLD, pearson_check_results=None):
    """
    Remove highly correlated features using appropriate methods.
    Pearson features are ONLY compared with other Pearson features.
    Spearman features are ONLY compared with other Spearman features.
    NO cross-correlations between groups.
    """
    print_section(f"REMOVING CORRELATED FEATURES (threshold={threshold})")
    
    # Separate features by distribution type
    pearson_features = []
    spearman_features = []
    
    for feat, results in pearson_check_results.items():
        if results['use_pearson']:
            pearson_features.append(feat)
        else:
            spearman_features.append(feat)
    
    print(f"\n📊 Using SEPARATE correlation approaches:")
    print(f"   Pearson correlation for: {len(pearson_features)} normal features")
    print(f"   Spearman correlation for: {len(spearman_features)} skewed features")
    print(f"   ⚠️ NO cross-correlation between groups")
    
    correlation_data = []
    
    # 1. Pearson correlations ONLY within Pearson group
    if len(pearson_features) > 1:
        print(f"\n   Computing Pearson correlations among {len(pearson_features)} features...")
        pearson_corr = X[pearson_features].corr(method='pearson').abs()
        upper_p = pearson_corr.where(np.triu(np.ones(pearson_corr.shape), k=1).astype(bool))
        
        for col in pearson_features:
            high_corr = upper_p[col][upper_p[col] > threshold]
            for corr_feat in high_corr.index:
                to_drop = decide_which_to_drop(X, col, corr_feat)
                correlation_data.append({
                    'feature1': col,
                    'feature2': corr_feat,
                    'correlation': upper_p[col][corr_feat],
                    'method': 'Pearson',
                    'dropping': to_drop,
                    'keeping': corr_feat if to_drop == col else col
                })
    
    # 2. Spearman correlations ONLY within Spearman group
    if len(spearman_features) > 1:
        print(f"   Computing Spearman correlations among {len(spearman_features)} features...")
        spearman_corr = X[spearman_features].corr(method='spearman').abs()
        upper_s = spearman_corr.where(np.triu(np.ones(spearman_corr.shape), k=1).astype(bool))
        
        for col in spearman_features:
            high_corr = upper_s[col][upper_s[col] > threshold]
            for corr_feat in high_corr.index:
                to_drop = decide_which_to_drop(X, col, corr_feat)
                correlation_data.append({
                    'feature1': col,
                    'feature2': corr_feat,
                    'correlation': upper_s[col][corr_feat],
                    'method': 'Spearman',
                    'dropping': to_drop,
                    'keeping': corr_feat if to_drop == col else col
                })
    
    # Sort by correlation value
    correlation_data = sorted(correlation_data, key=lambda x: x['correlation'], reverse=True)
    
    # Print correlated pairs with method used
    if correlation_data:
        print("\n🔗 TOP CORRELATED FEATURE PAIRS:")
        print("-" * 80)
        print(f"{'Feature 1':<25} {'Feature 2':<25} {'Corr':>6} {'Method':<10} {'Action':<15}")
        print("-" * 80)
        
        # Show top 20 pairs
        for item in correlation_data[:20]:
            f1 = item['feature1'][:24]
            f2 = item['feature2'][:24]
            corr = item['correlation']
            method = item['method']
            drop = item['dropping'][:14]
            
            # Mark priority features
            if any(kw in item['keeping'].lower() for kw in PRIORITY_KEYWORDS):
                drop = drop + " ⭐"
            
            print(f"{f1:<25} {f2:<25} {corr:>6.3f} {method:<10} Drop: {drop:<15}")
        
        if len(correlation_data) > 20:
            print(f"   ... and {len(correlation_data) - 20} more pairs")
        print("-" * 80)
    
    # Analyze which types of correlations were found
    method_counts = pd.Series([item['method'] for item in correlation_data]).value_counts()
    print(f"\n📊 Correlation types found:")
    for method, count in method_counts.items():
        print(f"   {method}: {count} pairs")
    
    # Get unique features to drop
    to_drop = list(set([item['dropping'] for item in correlation_data]))
    X_filtered = X.drop(columns=to_drop)
    
    # Show which features are being kept vs dropped by type
    print(f"\n📈 Feature retention by type:")
    pearson_kept = [f for f in pearson_features if f not in to_drop]
    pearson_dropped = [f for f in pearson_features if f in to_drop]
    spearman_kept = [f for f in spearman_features if f not in to_drop]
    spearman_dropped = [f for f in spearman_features if f in to_drop]
    
    print(f"   Normal features (Pearson): {len(pearson_kept)} kept, {len(pearson_dropped)} dropped")
    print(f"   Non-normal features (Spearman): {len(spearman_kept)} kept, {len(spearman_dropped)} dropped")
    
    # Check if priority features were kept
    priority_kept = [f for f in X_filtered.columns if any(kw in f.lower() for kw in PRIORITY_KEYWORDS)]
    if priority_kept:
        print(f"\n⭐ Priority features retained: {len(priority_kept)}")
        for feat in priority_kept[:5]:  # Show first 5
            print(f"   • {feat}")
        if len(priority_kept) > 5:
            print(f"   ... and {len(priority_kept) - 5} more")
    
    print(f"\n📊 Summary:")
    print(f"   Separate correlation approach used (no cross-group comparisons)")
    print(f"   Correlated pairs found: {len(correlation_data)}")
    print(f"   Features removed: {len(to_drop)}")
    print(f"   Features remaining: {len(X_filtered.columns)}")
    
    return X_filtered, to_drop, "separate_groups"

def select_features_with_cv_zones(X, y, max_features=MAX_FEATURES):
    """
    Select features using Random Forest importance with cross-validation stability.
    This operates on ZONE-LEVEL data (527 samples) for better statistical power.
    """
    print_section("FEATURE SELECTION WITH CROSS-VALIDATION (ZONE LEVEL)")
    
    print(f"   Using {len(X)} zones for feature importance calculation")
    print(f"   This provides better statistical power than using {168} stations")
    
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    importance_matrix = pd.DataFrame(index=X.columns)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        
        # Use zone-level RF parameters (can be less strict with 527 samples)
        rf = RandomForestRegressor(**RF_PARAMS_ZONE)
        rf.fit(X_train, y_train)
        
        importance_matrix[f'fold_{fold+1}'] = rf.feature_importances_
        
        print(f"   Fold {fold+1}: OOB R² = {rf.oob_score_:.3f} (train size: {len(X_train)})")
    
    # Calculate mean importance and stability
    importance_matrix['mean'] = importance_matrix.mean(axis=1)
    importance_matrix['std'] = importance_matrix.std(axis=1)
    importance_matrix['cv'] = importance_matrix['std'] / (importance_matrix['mean'] + 1e-10)
    importance_matrix['stability'] = 1 / (1 + importance_matrix['cv'])
    
    # Combined score
    importance_matrix['score'] = (
        IMPORTANCE_STABILITY_WEIGHT * importance_matrix['mean'] + 
        (1 - IMPORTANCE_STABILITY_WEIGHT) * importance_matrix['stability'] * importance_matrix['mean']
    )
    
    # Filter by minimum importance
    importance_matrix = importance_matrix[importance_matrix['mean'] >= MIN_IMPORTANCE]
    
    # Select top features
    n_features = min(max_features, len(importance_matrix), max(MIN_FEATURES, len(importance_matrix)))
    selected = importance_matrix.nlargest(n_features, 'score')
    
    print(f"\n✅ Selected {len(selected)} features based on zone-level importance")
    print(f"\nTop features by combined score:")
    for i, (feat, row) in enumerate(selected.head(30).iterrows(), 1):
        print(f"   {i:2d}. {feat[:80]:80s} | Importance: {row['mean']:.4f} | Stability: {row['stability']:.3f}")
    
    return selected.index.tolist(), importance_matrix

def aggregate_zones_to_stations(zone_data, station_coverage, station_list, selected_features):
    """
    Aggregate zone features to station level using coverage percentages.
    Only aggregates the selected features after correlation removal and importance selection.
    """
    print_section("AGGREGATING ZONES TO STATIONS")
    
    print(f"   Aggregating {len(selected_features)} selected features")
    
    station_data_list = []
    stations_included = []
    
    # Features to aggregate
    features_to_aggregate = selected_features + ['metro_trips']
    
    for station_id in station_list:
        if str(station_id) not in station_coverage:
            continue
            
        zones_covered = station_coverage[str(station_id)]
        if not zones_covered:
            continue
        
        station_features = {}
        weighted_trips = 0
        total_coverage = 0
        zones_found = 0
        
        # Aggregate across all zones covered by this station
        for zone_id_str, coverage_pct in zones_covered.items():
            zone_id = int(zone_id_str)
            zone_row = zone_data[zone_data['zona'] == zone_id]
            
            if zone_row.empty:
                continue
                
            zone_row = zone_row.iloc[0]
            weight = coverage_pct / 100.0
            zones_found += 1
            
            # Aggregate selected features only
            for col in selected_features:
                if col in zone_row.index:
                    if col not in station_features:
                        station_features[col] = 0
                    station_features[col] += zone_row[col] * weight
            
            # Aggregate metro trips
            if 'metro_trips' in zone_row.index:
                weighted_trips += zone_row['metro_trips'] * weight
            
            total_coverage += weight
        
        if zones_found > 0 and total_coverage > 0:
            station_features['station_id'] = station_id
            station_features['metro_trips'] = weighted_trips
            station_features['total_coverage'] = total_coverage
            station_features['zones_covered'] = zones_found
            station_data_list.append(station_features)
            stations_included.append(station_id)
    
    stations_df = pd.DataFrame(station_data_list)
    
    print(f"\n✅ Aggregated to {len(stations_df)} stations")
    print(f"\n📊 Coverage Statistics:")
    print(f"   Average total coverage: {stations_df['total_coverage'].mean():.2f}")
    print(f"   Average zones per station: {stations_df['zones_covered'].mean():.1f}")
    print(f"   Trip range: {stations_df['metro_trips'].min():.0f} - {stations_df['metro_trips'].max():.0f}")
    
    return stations_df, stations_included

def evaluate_final_features_stations(stations_df, selected_features):
    """
    Evaluate the selected features using cross-validation at STATION level.
    This is the final evaluation after feature selection and aggregation.
    """
    print_section("EVALUATING SELECTED FEATURES (STATION LEVEL)")
    
    X = stations_df[selected_features]
    y = stations_df['metro_trips']
    
    print(f"   Evaluating {len(selected_features)} features on {len(X)} stations")
    
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Use station-level RF parameters (stricter for 168 samples)
        rf = RandomForestRegressor(**RF_PARAMS_STATION)
        rf.fit(X_train, y_train)
        
        train_r2 = rf.score(X_train, y_train)
        test_r2 = rf.score(X_test, y_test)
        test_mae = mean_absolute_error(y_test, rf.predict(X_test))
        
        fold_results.append({
            'fold': fold,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'overfitting': train_r2 - test_r2
        })
        
        print(f"   Fold {fold}: Test R² = {test_r2:.3f}, Overfit = {train_r2 - test_r2:.3f}")
    
    results_df = pd.DataFrame(fold_results)
    
    print(f"\n📊 PERFORMANCE SUMMARY (Baseline for GraphSAGE):")
    print(f"   Test R²:     {results_df['test_r2'].mean():.3f} ± {results_df['test_r2'].std():.3f}")
    print(f"   Test MAE:    {results_df['test_mae'].mean():.0f} ± {results_df['test_mae'].std():.0f}")
    print(f"   Overfitting: {results_df['overfitting'].mean():.3f}")
    print(f"\n   Note: GraphSAGE should improve upon this by leveraging network structure")
    
    return results_df

def save_results(output_dir, zone_data, stations_df, selected_features, importance_df, 
                cv_results, dropped_features, correlation_method, pearson_check):
    """Save all results and create comprehensive report."""
    print_section("SAVING RESULTS")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save zone data with only selected important features
    important_zones_path = os.path.join(DADOS_DIR, 'dados_zonas_importantes.xlsx')
    important_columns = ['zona'] + selected_features + ['metro_trips']
    zone_data[important_columns].to_excel(important_zones_path, index=False)
    print(f"✅ Zone data with important features: {important_zones_path}")
    
    # Save feature importance results
    resultado_path = os.path.join(DADOS_DIR, 'resultado_dados_importantes.xlsx')
    # Create a clean dataframe with just feature name, importance, and stability
    resultado_df = importance_df[importance_df.index.isin(selected_features)][['mean', 'stability']].copy()
    resultado_df = resultado_df.rename(columns={'mean': 'Importance'})
    resultado_df = resultado_df.rename(columns={'stability': 'Stability'})
    resultado_df.index.name = 'Feature'
    resultado_df = resultado_df.sort_values('Importance', ascending=False)
    resultado_df.to_excel(resultado_path)
    print(f"✅ Feature importance results: {resultado_path}")
    
    # Save configuration and results
    features_path = os.path.join(output_dir, f'selected_features_{timestamp}.json')
    with open(features_path, 'w') as f:
        json.dump({
            'selected_features': selected_features,
            'dropped_correlated': dropped_features,
            'n_selected': len(selected_features),
            'correlation_method': correlation_method,
            'config': {
                'correlation_threshold': CORRELATION_THRESHOLD,
                'max_features': MAX_FEATURES,
                'n_folds': N_FOLDS,
                'importance_stability_weight': IMPORTANCE_STABILITY_WEIGHT,
                'skewness_threshold': SKEWNESS_THRESHOLD,
                'outlier_threshold': OUTLIER_THRESHOLD,
                'priority_keywords': PRIORITY_KEYWORDS
            }
        }, f, indent=2)
    print(f"✅ Features config: {features_path}")
    
    # Convert numpy types to Python native types for JSON serialization
    pearson_check_serializable = {}
    for key, value in pearson_check.items():
        pearson_check_serializable[key] = {
            'use_pearson': bool(value['use_pearson']),
            'skewness': float(value['skewness']),
            'outlier_pct': float(value['outlier_pct']),
            'unique_values': int(value['unique_values']),
            'is_normal': bool(value['is_normal']),
            'issues': value['issues']
        }
    
    # Save statistical validation results
    validation_path = os.path.join(output_dir, f'pearson_validation_{timestamp}.json')
    with open(validation_path, 'w') as f:
        json.dump(pearson_check_serializable, f, indent=2)
    print(f"✅ Validation results: {validation_path}")
    
    # Save other results
    importance_path = os.path.join(output_dir, f'feature_importance_{timestamp}.csv')
    importance_df.to_csv(importance_path)
    print(f"✅ Feature importance: {importance_path}")
    
    cv_path = os.path.join(output_dir, f'cv_baseline_{timestamp}.csv')
    cv_results.to_csv(cv_path, index=False)
    print(f"✅ CV baseline: {cv_path}")
    
    station_features_path = os.path.join(output_dir, f'station_features_{timestamp}.csv')
    stations_df[['station_id', 'metro_trips'] + selected_features].to_csv(station_features_path, index=False)
    print(f"✅ Station features: {station_features_path}")
    
    # Create detailed report
    report_path = os.path.join(output_dir, f'feature_selection_report_{timestamp}.txt')
    with open(report_path, 'w') as f:
        f.write("FEATURE SELECTION FOR GRAPHSAGE WITH STATISTICAL VALIDATION\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("METHODOLOGY:\n")
        f.write("-" * 40 + "\n")
        f.write("1. Statistical validation at zone level (527 samples)\n")
        f.write("2. Correlation removal using appropriate methods\n")
        f.write("3. Feature importance with Random Forest at zone level\n")
        f.write("4. Aggregation to stations only after feature selection\n")
        f.write("5. Final evaluation at station level (168 samples)\n\n")
        
        f.write("STATISTICAL VALIDATION:\n")
        f.write("-" * 40 + "\n")
        pearson_appropriate = sum(1 for r in pearson_check.values() if r['use_pearson'])
        f.write(f"Features appropriate for Pearson: {pearson_appropriate}/{len(pearson_check)}\n")
        f.write(f"Correlation method used: {correlation_method}\n\n")
        
        f.write("CONFIGURATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Correlation threshold: {CORRELATION_THRESHOLD}\n")
        f.write(f"Max features: {MAX_FEATURES}\n")
        f.write(f"Cross-validation folds: {N_FOLDS}\n")
        f.write(f"Skewness threshold: {SKEWNESS_THRESHOLD}\n")
        f.write(f"Outlier threshold: {OUTLIER_THRESHOLD}\n")
        f.write(f"Priority keywords: {', '.join(PRIORITY_KEYWORDS)}\n\n")
        
        f.write("RESULTS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Features removed (correlation): {len(dropped_features)}\n")
        f.write(f"Features selected: {len(selected_features)}\n")
        f.write(f"Baseline Test R²: {cv_results['test_r2'].mean():.3f} ± {cv_results['test_r2'].std():.3f}\n\n")
        
        f.write("SELECTED FEATURES:\n")
        f.write("-" * 40 + "\n")
        for i, feat in enumerate(selected_features, 1):
            f.write(f"{i:3d}. {feat}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("OUTPUT FILES CREATED:\n")
        f.write("-" * 40 + "\n")
        f.write(f"1. dados_zonas_importantes.xlsx - Zone data with selected features\n")
        f.write(f"2. resultado_dados_importantes.xlsx - Feature importance and stability\n")
        f.write(f"3. Files in {output_dir}/ with timestamp {timestamp}\n")
    
    print(f"✅ Report: {report_path}")
    
    return timestamp

def main():
    """Main execution pipeline with statistical validation at zone level."""
    print("=" * 70)
    print("FEATURE SELECTION PIPELINE")
    print("WITH ZONE-LEVEL STATISTICAL VALIDATION")
    print("=" * 70)
    
    # 1. Load ZONE data (527 samples)
    zone_data, station_coverage = load_data()
    
    # 2. Prepare zone features for analysis
    zone_feature_cols = [col for col in zone_data.columns 
                         if col not in ['zona', 'metro_trips']]
    X_zones = zone_data[zone_feature_cols]
    y_zones = zone_data['metro_trips']
    
    print(f"\nAnalyzing {len(X_zones)} zones with {len(zone_feature_cols)} features")
    
    # 3. CHECK PEARSON APPROPRIATENESS AT ZONE LEVEL (527 samples)
    print("\n🔍 Performing statistical validation at ZONE level...")
    pearson_check_results = check_pearson_appropriateness(X_zones, create_plots=True)
    
    # 4. Remove correlated features AT ZONE LEVEL using separate groups
    X_zones_filtered, dropped_features, correlation_method = remove_correlated_features_separate_groups(
        X_zones, pearson_check_results=pearson_check_results
    )
    
    print(f"\n✅ After correlation removal: {len(X_zones_filtered.columns)} features retained")
    
    # 5. FEATURE IMPORTANCE AT ZONE LEVEL (527 samples)
    # This is the key fix - using zones for RF importance, not stations
    selected_features, importance_df = select_features_with_cv_zones(
        X_zones_filtered, y_zones
    )
    
    print(f"\n✅ Final feature selection: {len(selected_features)} features")
    
    # 6. NOW aggregate to stations with only the final selected features
    active_stations = get_active_stations()
    print(f"\nActive stations: {len(active_stations)}")
    
    stations_df, station_ids = aggregate_zones_to_stations(
        zone_data, station_coverage, active_stations, selected_features
    )
    
    # 7. Evaluate performance at station level with selected features
    cv_results = evaluate_final_features_stations(stations_df, selected_features)
    
    # 8. Save results with validation info
    timestamp = save_results(
        OUTPUT_DIR, zone_data, stations_df, selected_features, 
        importance_df, cv_results, dropped_features,
        correlation_method, pearson_check_results
    )
    
    print(f"\n✅ Feature selection complete!")
    print(f"📂 Results saved to: {OUTPUT_DIR}")
    print(f"📅 Timestamp: {timestamp}")
    print(f"\n📊 Key outputs:")
    print(f"   1. dados_zonas_importantes.xlsx - Zone data with selected features")
    print(f"   2. resultado_dados_importantes.xlsx - Feature importance and stability") 
    print(f"   3. Use prepare_graphsage_data.py to prepare data for GraphSAGE")
    
    return {
        'selected_features': selected_features,
        'correlation_method': correlation_method,
        'baseline_r2': cv_results['test_r2'].mean(),
        'n_zones': len(zone_data),
        'n_stations': len(stations_df)
    }

if __name__ == "__main__":
    results = main()