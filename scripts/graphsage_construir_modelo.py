
import pickle
import numpy as np
import pandas as pd
import json
import sys
import os
import warnings
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from scipy import stats

warnings.filterwarnings('ignore')


# Model Parameters
HIDDEN_DIM = 64
DROPOUT = 0.2
LEARNING_RATE = 0.001
N_EPOCHS = 500
EARLY_STOPPING_PATIENCE = 50

# Data Split
N_FOLDS = 5
MONITOR_SPLIT = 0.2  # Use 20% of training data for early stopping monitoring

# Paths
sys.path.append('/Users/ellazyngier/Documents/github/tccII/scripts/dados')
sys.path.append('/Users/ellazyngier/Documents/github/tccII/scripts')
from dados.linhas import current_and_express_lines
from calculate_system_ridership import load_data as load_od_data

DATA_DIR = '/Users/ellazyngier/Documents/github/tccII/scripts/resultados'
DADOS_DIR = '/Users/ellazyngier/Documents/github/tccII/scripts/dados'
OUTPUT_DIR = '/Users/ellazyngier/Documents/github/tccII/scripts/resultados/expansion'

os.makedirs(OUTPUT_DIR, exist_ok=True)


def print_section(title, char="="):
    """Print formatted section header."""
    print("\n" + char*70)
    print(title)
    print(char*70)



def load_8_features():
    """Load the 8 selected features from Excel file."""
    print_section("LOADING 8 SELECTED FEATURES")
    
    # Load the Excel file with 8 features
    features_path = os.path.join(DADOS_DIR, '8_features.xlsx')
    features_df = pd.read_excel(features_path)
    
    # Get feature names (all columns except 'zona')
    feature_names = [col for col in features_df.columns if col != 'zona']
    
    print(f"✅ Loaded {len(features_df)} zones with {len(feature_names)} features")
    print("\n📊 Features loaded:")
    for i, feat in enumerate(feature_names, 1):
        print(f"   {i}. {feat}")
    
    return features_df, feature_names

def aggregate_zones_to_stations(features_df, feature_names):
    """
    Aggregate zone features to station level using coverage percentages.
    Simple weighted sum based on coverage.
    """
    print_section("AGGREGATING ZONES TO STATIONS")
    
    # Load station-zone coverage
    coverage_path = os.path.join(DADOS_DIR, 'station_zones_882.json')
    with open(coverage_path, 'r') as f:
        station_zones = json.load(f)
    
    print(f"✅ Loaded coverage for {len(station_zones)} stations")
    
    # Get active stations
    active_stations = set()
    for line in current_and_express_lines:
        active_stations.update(line)
    
    # Convert to list and ensure consistent type
    station_ids = sorted([int(s) for s in active_stations if str(s) in station_zones])
    
    print(f"   Processing {len(station_ids)} active stations with coverage data")
    
    # Aggregate features for each station
    station_features = []
    
    for station_id in tqdm(station_ids, desc="   Aggregating to stations"):
        # Get zone coverages for this station
        zone_coverages = station_zones[str(station_id)]
        
        # Calculate weighted sum of features
        weighted_features = np.zeros(len(feature_names))
        
        for zone_id, coverage_pct in zone_coverages.items():
            zone_int = int(zone_id)
            zone_row = features_df[features_df['zona'] == zone_int]
            
            if len(zone_row) > 0:
                # Get feature values for this zone
                zone_values = zone_row.iloc[0][feature_names].values
                
                # Add weighted contribution
                weight = coverage_pct / 100.0
                weighted_features += zone_values * weight
        
        station_features.append(weighted_features)
    
    # Convert to numpy array
    X = np.array(station_features)
    
    print(f"✅ Created feature matrix: {X.shape}")
    print(f"   Stations: {len(station_ids)}")
    print(f"   Features: {len(feature_names)}")
    
    # Show feature statistics
    print("\n📊 Aggregated feature statistics:")
    for i, name in enumerate(feature_names):
        values = X[:, i]
        print(f"   {name[:80]:80s}: mean={values.mean():10.1f}, std={values.std():10.1f}")
    
    return X, station_ids, station_zones

def calculate_correct_ridership(station_ids, station_zones_data, travel_matrix):
    """
    Calculate CORRECT ridership (only metro-to-metro trips).
    """
    print_section("CALCULATING METRO RIDERSHIP")
    
    y = []
    
    for sid in tqdm(station_ids, desc="   Computing ridership"):
        sid_str = str(sid)
        
        if sid_str not in station_zones_data:
            y.append(0)
            continue
        
        station_zones = station_zones_data[sid_str]
        station_ridership = 0
        
        # Count trips to/from all other stations (metro-to-metro only)
        for other_sid in station_ids:
            other_sid_str = str(other_sid)
            if other_sid_str not in station_zones_data:
                continue
                
            other_zones = station_zones_data[other_sid_str]
            
            # Calculate flow between these two stations
            for zone_a, pct_a in station_zones.items():
                for zone_b, pct_b in other_zones.items():
                    zone_a_idx = int(zone_a) - 1
                    zone_b_idx = int(zone_b) - 1
                    
                    if 0 <= zone_a_idx < 527 and 0 <= zone_b_idx < 527:
                        if sid == other_sid and zone_a_idx >= zone_b_idx:
                            continue  # Avoid double counting within same station
                        
                        # Trips in both directions
                        trips_ab = travel_matrix[zone_a_idx, zone_b_idx]
                        trips_ba = travel_matrix[zone_b_idx, zone_a_idx]
                        
                        flow = (trips_ab + trips_ba) * (pct_a/100) * (pct_b/100)
                        station_ridership += flow
        
        y.append(station_ridership)
    
    y = np.array(y)
    
    print(f"✅ Calculated ridership for {len(y)} stations")
    print(f"   Mean: {y.mean():.0f} trips/day")
    print(f"   Std: {y.std():.0f}")
    print(f"   Min: {y.min():.0f}")
    print(f"   Max: {y.max():.0f}")
    print(f"   Zero ridership stations: {(y == 0).sum()}")
    
    return y

def create_metro_graph(station_ids):
    """Create graph structure from metro lines."""
    print_section("CREATING METRO GRAPH")
    
    edges = []
    
    for line in current_and_express_lines:
        for i in range(len(line) - 1):
            if line[i] in station_ids and line[i+1] in station_ids:
                idx1 = station_ids.index(line[i])
                idx2 = station_ids.index(line[i+1])
                edges.append([idx1, idx2])
                edges.append([idx2, idx1])
    
    edges = list(set(map(tuple, edges)))
    edge_index = np.array(edges).T if edges else np.array([[], []])
    
    print(f"✅ Created graph:")
    print(f"   Nodes: {len(station_ids)}")
    print(f"   Edges: {edge_index.shape[1]//2 if edge_index.size > 0 else 0}")
    
    return edge_index


class GraphSAGEModel(nn.Module):
    """GraphSAGE model with 8 input features."""
    
    def __init__(self, num_features=8, hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        self.conv1 = SAGEConv(num_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim // 2)
        self.linear = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.linear(x)
        return x.squeeze()

def detect_fitting_status(train_r2, test_r2):
    """
    Simplified fitting detection based on train-test gap.
    """
    gap = train_r2 - test_r2
    
    if train_r2 < 0.3 and test_r2 < 0.3:
        status = "underfitting"
        message = f"Model is UNDERFITTING (train R²={train_r2:.3f}, test R²={test_r2:.3f})"
    elif gap > 0.2:
        status = "overfitting"
        message = f"Model shows OVERFITTING (train-test gap={gap:.3f})"
    elif test_r2 > 0.7 and gap < 0.1:
        status = "excellent"
        message = f"Model shows EXCELLENT GENERALIZATION (test R²={test_r2:.3f}, gap={gap:.3f})"
    elif test_r2 > 0.5:
        status = "good"
        message = f"Model shows GOOD PERFORMANCE (test R²={test_r2:.3f})"
    else:
        status = "acceptable"
        message = f"Model performance is ACCEPTABLE (test R²={test_r2:.3f})"
    
    return status, message

def train_final_model(X, y, edge_index, feature_names):
    """
    Train final model on ALL data for deployment.
    """
    print_section("TRAINING FINAL MODEL ON ALL DATA")
    
    # Normalize all features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convert to PyTorch
    x = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
    
    # Create final model
    final_model = GraphSAGEModel(num_features=X.shape[1])
    optimizer = torch.optim.Adam(final_model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    # Train on all data
    print("Training final model...")
    for epoch in range(N_EPOCHS):
        final_model.train()
        optimizer.zero_grad()
        out = final_model(x, edge_index_tensor)
        loss = criterion(out, y_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"   Epoch {epoch}: Loss={loss.item():.4f}")
    
    # Evaluate final model
    final_model.eval()
    with torch.no_grad():
        predictions = final_model(x, edge_index_tensor)
        final_r2 = r2_score(y, predictions.numpy())
        final_mae = mean_absolute_error(y, predictions.numpy())
    
    print(f"\n✅ Final model performance on full dataset:")
    print(f"   R²: {final_r2:.3f}")
    print(f"   MAE: {final_mae:.0f}")
    
    # Save model and scaler
    model_path = '/Users/ellazyngier/Documents/github/tccII/scripts/dados/graphsage_model.pt'
    scaler_path = '/Users/ellazyngier/Documents/github/tccII/scripts/dados/scaler.pkl'
    
    torch.save(final_model.state_dict(), model_path)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"\n✅ Model saved to: {model_path}")
    print(f"✅ Scaler saved to: {scaler_path}")
    
    return final_model, scaler


def run_simple_kfold(X, y, edge_index, station_ids, feature_names, n_folds=N_FOLDS):
    """
    Run simple k-fold cross-validation without separate validation set.
    Uses a small monitoring subset from training data for early stopping.
    """
    print_section(f"{n_folds}-FOLD CROSS-VALIDATION")
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_results = []
    all_predictions = {}
    all_actuals = {}
    
    print(f"\n🔄 Running {n_folds}-fold cross-validation...")
    print("-" * 70)
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        print(f"\n📊 FOLD {fold}/{n_folds}")
        
        # Split training data for early stopping monitoring (optional)
        # We use a small subset of training data to monitor convergence
        n_monitor = int(MONITOR_SPLIT * len(train_idx))
        monitor_idx = train_idx[:n_monitor]
        actual_train_idx = train_idx[n_monitor:]
        
        print(f"   Train: {len(actual_train_idx)}, Monitor: {len(monitor_idx)}, Test: {len(test_idx)}")
        
        # Create masks
        n_nodes = len(X)
        train_mask = np.zeros(n_nodes, dtype=bool)
        monitor_mask = np.zeros(n_nodes, dtype=bool)
        test_mask = np.zeros(n_nodes, dtype=bool)
        
        train_mask[actual_train_idx] = True
        monitor_mask[monitor_idx] = True
        test_mask[test_idx] = True
        
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X[train_idx])  # Fit on all training data
        X_scaled_full = scaler.transform(X)  # Transform all data
        
        # Convert to PyTorch
        x = torch.tensor(X_scaled_full, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
        
        # Create and train model
        model = GraphSAGEModel(num_features=X.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.MSELoss()
        
        best_monitor_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        best_epoch = 0
        
        # Training
        for epoch in range(N_EPOCHS):
            model.train()
            optimizer.zero_grad()
            out = model(x, edge_index_tensor)
            
            # Train on actual training set
            train_loss = criterion(out[train_mask], y_tensor[train_mask])
            train_loss.backward()
            optimizer.step()
            
            # Monitor on monitoring set for early stopping
            model.eval()
            with torch.no_grad():
                out = model(x, edge_index_tensor)
                monitor_loss = criterion(out[monitor_mask], y_tensor[monitor_mask])
            
            # Early stopping based on monitor set
            if monitor_loss < best_monitor_loss:
                best_monitor_loss = monitor_loss
                patience_counter = 0
                best_model_state = model.state_dict()
                best_epoch = epoch
            else:
                patience_counter += 1
            
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                break
            
            # Print progress occasionally
            if epoch % 100 == 0:
                print(f"      Epoch {epoch}: Loss={train_loss.item():.4f}")
        
        print(f"      Training stopped at epoch {best_epoch}")
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            predictions = model(x, edge_index_tensor)
        
        # Calculate metrics - using all training data for final train score
        y_train_all = y_tensor[train_idx].numpy()
        y_test = y_tensor[test_idx].numpy()
        
        pred_train_all = predictions[train_idx].numpy()
        pred_test = predictions[test_idx].numpy()
        
        train_r2 = r2_score(y_train_all, pred_train_all)
        test_r2 = r2_score(y_test, pred_test)
        
        train_mae = mean_absolute_error(y_train_all, pred_train_all)
        test_mae = mean_absolute_error(y_test, pred_test)
        
        train_rmse = np.sqrt(mean_squared_error(y_train_all, pred_train_all))
        test_rmse = np.sqrt(mean_squared_error(y_test, pred_test))
        
        print(f"   Results - Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f}, Gap: {train_r2-test_r2:.3f}")
        
        # Store results
        fold_results.append({
            'fold': fold,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_test_gap': train_r2 - test_r2
        })
        
        # Store predictions for analysis
        for idx in test_idx:
            if station_ids[idx] not in all_predictions:
                all_predictions[station_ids[idx]] = []
                all_actuals[station_ids[idx]] = y[idx]
            all_predictions[station_ids[idx]].append(predictions[idx].item())
    
    print("-" * 70)
    
    # Aggregate results
    results_df = pd.DataFrame(fold_results)
    
    print_section("CROSS-VALIDATION SUMMARY")
    
    print("\n📊 PERFORMANCE ACROSS FOLDS:")
    print(f"   {'Metric':<15} {'Mean':>8} ± {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 55)
    
    for metric in ['train_r2', 'test_r2', 'train_mae', 'test_mae']:
        mean_val = results_df[metric].mean()
        std_val = results_df[metric].std()
        min_val = results_df[metric].min()
        max_val = results_df[metric].max()
        
        metric_name = metric.replace('_', ' ').title().replace('R2', 'R²')
        if 'mae' in metric:
            print(f"   {metric_name:<15} {mean_val:>8.0f} ± {std_val:>8.0f} {min_val:>8.0f} {max_val:>8.0f}")
        else:
            print(f"   {metric_name:<15} {mean_val:>8.3f} ± {std_val:>8.3f} {min_val:>8.3f} {max_val:>8.3f}")
    
    print("\n📊 GENERALIZATION:")
    mean_gap = results_df['train_test_gap'].mean()
    print(f"   Mean Train-Test Gap: {mean_gap:.3f}")
    
    # Fitting analysis
    mean_train = results_df['train_r2'].mean()
    mean_test = results_df['test_r2'].mean()
    
    status, message = detect_fitting_status(mean_train, mean_test)
    
    print(f"\n🎯 MODEL ASSESSMENT:")
    print(f"   {message}")
    
    # Confidence intervals
    confidence = 0.95
    test_scores = results_df['test_r2'].values
    mean_test = np.mean(test_scores)
    se_test = stats.sem(test_scores)
    ci = stats.t.interval(confidence, len(test_scores)-1, loc=mean_test, scale=se_test)
    
    print(f"\n📈 Test R² Confidence Interval (95%):")
    print(f"   {ci[0]:.3f} to {ci[1]:.3f}")
    
    # Find stations with high prediction variance
    print("\n🔍 STATIONS WITH HIGH PREDICTION VARIANCE:")
    station_variances = []
    for station_id, preds in all_predictions.items():
        if len(preds) > 1:  # Station was in test set multiple times
            variance = np.var(preds)
            actual = all_actuals[station_id]
            station_variances.append((station_id, actual, np.mean(preds), variance))
    
    if station_variances:
        station_variances.sort(key=lambda x: x[3], reverse=True)
        print("   Station | Actual | Mean Pred | Variance")
        print("   --------|--------|-----------|----------")
        for sid, actual, mean_pred, var in station_variances[:5]:
            print(f"   {sid:7} | {actual:6.0f} | {mean_pred:9.0f} | {var:9.0f}")
    
    return results_df, all_predictions, all_actuals, status

def create_visualization(cv_results, X, feature_names, station_ids):
    """Create comprehensive visualization of results."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Box plots of R² scores
    train_test_data = [cv_results['train_r2'].values, cv_results['test_r2'].values]
    bp = axes[0, 0].boxplot(train_test_data, labels=['Train', 'Test'], patch_artist=True)
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].set_title('Performance Distribution')
    axes[0, 0].axhline(y=0.5, color='g', linestyle='--', alpha=0.3, label='Good threshold')
    axes[0, 0].axhline(y=0.7, color='b', linestyle='--', alpha=0.3, label='Excellent threshold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].legend()
    
    # 2. Fold-by-fold performance
    folds = cv_results['fold'].values
    axes[0, 1].plot(folds, cv_results['train_r2'], 'o-', label='Train', alpha=0.7, markersize=8)
    axes[0, 1].plot(folds, cv_results['test_r2'], 's-', label='Test', alpha=0.7, markersize=8)
    axes[0, 1].fill_between(folds, cv_results['train_r2'], cv_results['test_r2'], alpha=0.2)
    axes[0, 1].set_xlabel('Fold')
    axes[0, 1].set_ylabel('R² Score')
    axes[0, 1].set_title('Performance by Fold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xticks(folds)
    
    # 3. Train-Test Gap Analysis
    gaps = cv_results['train_test_gap'].values
    bars = axes[0, 2].bar(folds, gaps, alpha=0.7, color=['green' if g < 0.1 else 'orange' if g < 0.2 else 'red' for g in gaps])
    axes[0, 2].set_xlabel('Fold')
    axes[0, 2].set_ylabel('Train-Test Gap')
    axes[0, 2].set_title('Generalization Gap by Fold')
    axes[0, 2].axhline(y=0.1, color='g', linestyle='--', alpha=0.5, label='Good (<0.1)')
    axes[0, 2].axhline(y=0.2, color='r', linestyle='--', alpha=0.5, label='Concern (>0.2)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    axes[0, 2].set_xticks(folds)
    
    # 4. Error metrics comparison
    mae_data = [cv_results['train_mae'].values, cv_results['test_mae'].values]
    bp2 = axes[1, 0].boxplot(mae_data, labels=['Train', 'Test'], patch_artist=True)
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
    axes[1, 0].set_ylabel('MAE (trips/day)')
    axes[1, 0].set_title('Mean Absolute Error')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 5. Feature statistics
    feature_stds = X.std(axis=0)
    feature_cvs = feature_stds / (X.mean(axis=0) + 1e-10)  # Coefficient of variation
    axes[1, 1].barh(range(len(feature_names)), feature_cvs)
    axes[1, 1].set_yticks(range(len(feature_names)))
    axes[1, 1].set_yticklabels([f[:25] + '...' if len(f) > 25 else f for f in feature_names], fontsize=8)
    axes[1, 1].set_xlabel('Coefficient of Variation')
    axes[1, 1].set_title('Feature Variability')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    # 6. Summary text
    mean_train = cv_results['train_r2'].mean()
    std_train = cv_results['train_r2'].std()
    mean_test = cv_results['test_r2'].mean()
    std_test = cv_results['test_r2'].std()
    
    # Calculate 95% CI
    confidence = 0.95
    test_scores = cv_results['test_r2'].values
    se_test = stats.sem(test_scores)
    ci = stats.t.interval(confidence, len(test_scores)-1, loc=mean_test, scale=se_test)
    
    summary_text = f"""
    K-FOLD CROSS-VALIDATION RESULTS
    ════════════════════════════════
    
    Performance (mean ± std):
    • Train R²: {mean_train:.3f} ± {std_train:.3f}
    • Test R²:  {mean_test:.3f} ± {std_test:.3f}
    
    Test R² 95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]
    
    Generalization:
    • Train-Test Gap: {cv_results['train_test_gap'].mean():.3f}
    • Gap Std Dev: {cv_results['train_test_gap'].std():.3f}
    
    Error Metrics:
    • Test MAE: {cv_results['test_mae'].mean():.0f} ± {cv_results['test_mae'].std():.0f}
    • Test RMSE: {cv_results['test_rmse'].mean():.0f} ± {cv_results['test_rmse'].std():.0f}
    
    Configuration:
    • Features: {len(feature_names)}
    • Folds: {len(cv_results)}
    • Stations: {len(station_ids)}
    • Hidden Dim: {HIDDEN_DIM}
    • Learning Rate: {LEARNING_RATE}
    """
    
    axes[1, 2].text(0.05, 0.5, summary_text, fontsize=9, family='monospace',
                   verticalalignment='center', transform=axes[1, 2].transAxes)
    axes[1, 2].axis('off')
    
    plt.suptitle('GraphSAGE with 8 Features - Simple K-Fold Cross-Validation', fontsize=14, y=1.02)
    plt.tight_layout()
    
    return fig

def main():
    """Main execution pipeline with simple k-fold cross-validation."""
    print("=" * 70)
    print("GRAPHSAGE WITH SIMPLE K-FOLD CROSS-VALIDATION")
    print("=" * 70)
    print("\n📌 Configuration:")
    print(f"   • {N_FOLDS}-fold cross-validation")
    print(f"   • No separate validation set")
    print(f"   • {MONITOR_SPLIT*100:.0f}% of training used for early stopping")
    print("   • Clean train-test evaluation")
    
    # 1. Load the 8 features
    features_df, feature_names = load_8_features()
    
    # 2. Aggregate zones to stations
    X, station_ids, station_zones = aggregate_zones_to_stations(features_df, feature_names)
    
    # 3. Load OD matrix and calculate ridership
    print("\n   Loading OD matrix...")
    _, travel_matrix = load_od_data()
    y = calculate_correct_ridership(station_ids, station_zones, travel_matrix)
    
    # 4. Create metro graph
    edge_index = create_metro_graph(station_ids)
    
    # 5. Run simple k-fold cross-validation
    cv_results, all_predictions, all_actuals, status = run_simple_kfold(
        X, y, edge_index, station_ids, feature_names, n_folds=N_FOLDS
    )
    
    # 6. Train final model on all data and save it
    final_model, scaler = train_final_model(X, y, edge_index, feature_names)
    
    # 7. Create visualization
    fig = create_visualization(cv_results, X, feature_names, station_ids)
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(OUTPUT_DIR, f'graphsage_simple_kfold_{timestamp}.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Analysis saved to: {plot_path}")
    
    plt.show()
    
    # Final reporting
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    mean_test = cv_results['test_r2'].mean()
    std_test = cv_results['test_r2'].std()
    ci = stats.t.interval(0.95, len(cv_results)-1, loc=mean_test, scale=stats.sem(cv_results['test_r2']))
    
    print(f"\n📊 THESIS REPORTING:")
    print(f"   'The GraphSAGE model achieved R² = {mean_test:.3f} ± {std_test:.3f}'")
    print(f"   'using {N_FOLDS}-fold cross-validation (95% CI: [{ci[0]:.3f}, {ci[1]:.3f}])'")
    
    if status == "excellent":
        print(f"\n   ✅ The model demonstrates excellent generalization to unseen data!")
    elif status == "good":
        print(f"\n   ✅ The model shows good performance with reasonable generalization.")
    
    return {
        'cv_results': cv_results,
        'mean_test_r2': mean_test,
        'std_test_r2': std_test,
        'ci_95': ci,
        'status': status,
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'n_stations': len(station_ids)
    }

if __name__ == "__main__":
    results = main()