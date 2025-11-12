
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.validation import make_valid
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import List, Tuple, Dict, Optional, Set
from tqdm import tqdm
import time
import os
import sys
from datetime import datetime
from copy import deepcopy
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial
import hashlib

# Suppress shapely warnings
warnings.filterwarnings('ignore', message='.*initial implementation of Parquet.*')
warnings.filterwarnings('ignore', message='.*Geometry is in a geographic CRS.*')

# Add paths
sys.path.append('/Users/ellazyngier/Documents/github/tccII/scripts/dados')
sys.path.append('/Users/ellazyngier/Documents/github/tccII/scripts')
from dados.linhas import current_and_express_lines

# Paths
DATA_DIR = '/Users/ellazyngier/Documents/github/tccII/scripts/resultados'
DADOS_DIR = '/Users/ellazyngier/Documents/github/tccII/scripts/dados'
OUTPUT_DIR = '/Users/ellazyngier/Documents/github/tccII/scripts/resultados/expansion'
MODEL_PATH = '/Users/ellazyngier/Documents/github/tccII/scripts/dados/graphsage_model.pt'
SCALER_PATH = '/Users/ellazyngier/Documents/github/tccII/scripts/dados/scaler.pkl'
ZONES_SHAPEFILE_PATH = '/Users/ellazyngier/Documents/github/tccII/Site_190225/002_Site Metro Mapas_190225/Shape/Zonas_2023.shp'

os.makedirs(OUTPUT_DIR, exist_ok=True)

class GraphSAGEModel(nn.Module):
    """GraphSAGE model with 8 input features."""
    
    def __init__(self, num_features=8, hidden_dim=64, dropout=0.2):
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


def evaluate_candidate_batch(candidate_batch_data):
    """
    Evaluate a batch of candidates in parallel.
    This function is designed to be picklable for multiprocessing.
    """
    (candidates, current_stations, current_connections, 
     existing_stations, min_distance, max_distance, 
     model_path, scaler_path, features_df, feature_names,
     existing_features, line_connections, zones_gdf,
     catchment_radius, existing_station_zones) = candidate_batch_data
    
    # Load model and scaler in worker process
    model = GraphSAGEModel(num_features=8)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    results = []
    
    for lat, lon in candidates:
        all_current_stations = existing_stations + current_stations
        new_idx = len(all_current_stations)
        
        # Find valid connections
        valid_connections = []
        for idx, (station_lat, station_lon) in enumerate(all_current_stations):
            dist = calculate_distance_simple(lat, lon, station_lat, station_lon)
            if min_distance <= dist <= max_distance:
                valid_connections.append((idx, dist))
        
        if not valid_connections:
            continue
        
        # Sort by distance and try best connections
        valid_connections.sort(key=lambda x: x[1])
        
        for conn_idx, _ in valid_connections[:2]:  # Try top 2 connections
            new_stations = current_stations + [(lat, lon)]
            new_connections = current_connections + [(conn_idx, new_idx)]
            
            # Evaluate this configuration
            try:
                ridership = evaluate_network_simple(
                    new_stations, new_connections, existing_stations,
                    existing_features, line_connections, scaler, model,
                    features_df, feature_names, zones_gdf, catchment_radius,
                    existing_station_zones
                )
                
                if ridership > 0:
                    results.append((new_stations, new_connections, ridership))
            except:
                continue
    
    return results

def calculate_distance_simple(lat1, lon1, lat2, lon2):
    """Simple distance calculation for parallelization."""
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * \
        np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

def evaluate_network_simple(new_stations, new_connections, existing_stations,
                           existing_features, line_connections, scaler, model,
                           features_df, feature_names, zones_gdf, catchment_radius,
                           existing_station_zones):
    """Simplified network evaluation for parallel processing."""
    try:
        # Calculate features for new stations
        all_features = [existing_features]
        
        for i, (lat, lon) in enumerate(new_stations):
            existing_for_overlap = existing_stations + new_stations[:i]
            features = calculate_station_features_simple(
                lat, lon, existing_for_overlap, features_df, 
                feature_names, zones_gdf, catchment_radius
            )
            all_features.append(features.reshape(1, -1))
        
        if len(all_features) > 1:
            X = np.vstack(all_features)
        else:
            X = all_features[0]
        
        # Build edge index
        edges = []
        for idx1, idx2, _ in line_connections:
            edges.append([idx1, idx2])
            edges.append([idx2, idx1])
        
        for from_idx, to_idx in new_connections:
            edges.append([from_idx, to_idx])
            edges.append([to_idx, from_idx])
        
        edge_index = np.array(edges).T if edges else np.array([[], []])
        
        # Normalize and predict
        X_scaled = scaler.transform(X)
        x = torch.tensor(X_scaled, dtype=torch.float32)
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
        
        with torch.no_grad():
            predictions = model(x, edge_index_tensor)
        
        return predictions.sum().item()
    except:
        return 0.0

def calculate_station_features_simple(lat, lon, existing_stations, features_df, 
                                     feature_names, zones_gdf, catchment_radius):
    """Simplified feature calculation for parallel processing."""
    try:
        # Simple zone coverage without overlap handling for speed
        station_point = Point(lon, lat)
        station_gdf = gpd.GeoDataFrame([{'id': 'NEW'}], 
                                       geometry=[station_point],
                                       crs='EPSG:4326')
        
        station_projected = station_gdf.to_crs('EPSG:32723')
        zones_projected = zones_gdf.to_crs('EPSG:32723')
        
        station_buffer = station_projected.geometry[0].buffer(catchment_radius)
        
        weighted_features = np.zeros(len(feature_names))
        
        for idx, zone in zones_projected.iterrows():
            try:
                intersection = station_buffer.intersection(zone.geometry)
                if not intersection.is_empty:
                    coverage_pct = (intersection.area / zone.geometry.area) * 100
                    if coverage_pct > 0.1:
                        zone_id = idx + 1
                        zone_row = features_df[features_df['zona'] == zone_id]
                        
                        if len(zone_row) > 0:
                            zone_values = zone_row.iloc[0][feature_names].values
                            weight = coverage_pct / 100.0
                            weighted_features += zone_values * weight
            except:
                continue
        
        return weighted_features
    except:
        return np.zeros(len(feature_names))


class GraphSAGENetworkExpander:
    def __init__(self, min_distance: float = 1.0, max_distance: float = 2.1, 
                 catchment_radius: int = 882, use_parallel: bool = True):
        """
        Initialize network expander with trained GraphSAGE model.
        
        Args:
            min_distance: Minimum distance between stations (km)
            max_distance: Maximum distance between stations (km)
            catchment_radius: Station catchment radius in meters
            use_parallel: Whether to use parallel processing
        """
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.catchment_radius = catchment_radius
        self.overlap_check_distance = (catchment_radius * 2) / 1000  # 1.764 km
        self.use_parallel = use_parallel
        
        # Initialize caches
        self.zone_cache = {}  # Cache for zone calculations
        self.distance_cache = {}  # Cache for distance calculations
        self.feature_cache = {}  # Cache for feature calculations
        
        print("="*70)
        print("OPTIMIZED GRAPHSAGE NETWORK EXPANDER")
        print("="*70)
        print(f"Configuration:")
        print(f"  Min distance: {min_distance} km")
        print(f"  Max distance: {max_distance} km")
        print(f"  Catchment radius: {catchment_radius} m")
        print(f"  Parallel processing: {use_parallel}")
        print(f"  CPU cores available: {cpu_count()}")
        
        # Load model and scaler
        print("\n📊 Loading trained model...")
        self.model = GraphSAGEModel(num_features=8)
        self.model.load_state_dict(torch.load(MODEL_PATH))
        self.model.eval()
        print("✅ Model loaded")
        
        with open(SCALER_PATH, 'rb') as f:
            self.scaler = pickle.load(f)
        print("✅ Scaler loaded")
        
        # Load data
        print("\n📊 Loading network data...")
        self._load_existing_network()
        self._load_zone_data()
        self._load_features()
        
        # Initialize expansion tracking
        self.added_stations = []
        self.added_connections = []
        self.baseline_ridership = None
        self.final_ridership = None
        self.total_expansion_time = 0
        
        print(f"✅ Cache initialized (will store up to 10,000 calculations)")
        
    def _load_existing_network(self):
        """Load existing stations and network structure."""
        df = pd.read_csv(os.path.join(DADOS_DIR, 'estacoes.csv'))
        
        active_stations = set()
        for line in current_and_express_lines:
            active_stations.update(line)
        
        with open(os.path.join(DADOS_DIR, 'station_zones_882.json'), 'r') as f:
            station_zones = json.load(f)
        
        self.station_ids = sorted([int(s) for s in active_stations if str(s) in station_zones])
        
        self.existing_stations = []
        for sid in self.station_ids:
            row = df[df['codigo_estacao'] == sid]
            if not row.empty:
                self.existing_stations.append((row.iloc[0]['latitude'], row.iloc[0]['longitude']))
        
        print(f"✅ Loaded {len(self.existing_stations)} existing stations")
        
        self.line_connections = []
        for line in current_and_express_lines:
            for i in range(len(line) - 1):
                if line[i] in self.station_ids and line[i+1] in self.station_ids:
                    idx1 = self.station_ids.index(line[i])
                    idx2 = self.station_ids.index(line[i+1])
                    self.line_connections.append((idx1, idx2, 'gray'))
    
    def _load_zone_data(self):
        """Load zone shapefile and coverage data."""
        self.zones_gdf = gpd.read_file(ZONES_SHAPEFILE_PATH)
        self.zones_gdf['geometry'] = self.zones_gdf['geometry'].apply(lambda geom: make_valid(geom))
        print(f"✅ Loaded {len(self.zones_gdf)} zones")
        
        with open(os.path.join(DADOS_DIR, 'station_zones_882.json'), 'r') as f:
            self.existing_station_zones = json.load(f)
    
    def _load_features(self):
        """Load the 8 features for zones."""
        features_path = os.path.join(DADOS_DIR, '8_features.xlsx')
        self.features_df = pd.read_excel(features_path)
        self.feature_names = [col for col in self.features_df.columns if col != 'zona']
        print(f"✅ Loaded {len(self.feature_names)} features")
        
        self.existing_features = self._aggregate_zones_to_stations(self.station_ids)
    
    def _aggregate_zones_to_stations(self, station_ids):
        """Aggregate zone features to station level."""
        station_features = []
        
        for station_id in station_ids:
            zone_coverages = self.existing_station_zones[str(station_id)]
            weighted_features = np.zeros(len(self.feature_names))
            
            for zone_id, coverage_pct in zone_coverages.items():
                zone_int = int(zone_id)
                zone_row = self.features_df[self.features_df['zona'] == zone_int]
                
                if len(zone_row) > 0:
                    zone_values = zone_row.iloc[0][self.feature_names].values
                    weight = coverage_pct / 100.0
                    weighted_features += zone_values * weight
            
            station_features.append(weighted_features)
        
        return np.array(station_features)
    
    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance with caching."""
        # Create cache key
        cache_key = (round(lat1, 6), round(lon1, 6), round(lat2, 6), round(lon2, 6))
        
        if cache_key in self.distance_cache:
            return self.distance_cache[cache_key]
        
        # Calculate if not cached
        R = 6371
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * \
            np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
        dist = R * 2 * np.arcsin(np.sqrt(a))
        
        # Cache result (limit cache size)
        if len(self.distance_cache) < 10000:
            self.distance_cache[cache_key] = dist
        
        return dist
    
    def get_station_zones_with_overlaps(self, lat: float, lon: float, 
                                       existing_station_coords: List[Tuple[float, float]]) -> Dict[str, float]:
        """Calculate zone coverage with caching."""
        # Create cache key
        cache_key = (round(lat, 4), round(lon, 4), 
                    tuple((round(la, 4), round(lo, 4)) for la, lo in existing_station_coords[:5]))
        
        if cache_key in self.zone_cache:
            return self.zone_cache[cache_key]
        
        try:
            # Calculate zone coverage (simplified for speed)
            station_point = Point(lon, lat)
            station_gdf = gpd.GeoDataFrame([{'id': 'NEW'}], 
                                           geometry=[station_point],
                                           crs='EPSG:4326')
            
            station_projected = station_gdf.to_crs('EPSG:32723')
            zones_projected = self.zones_gdf.to_crs('EPSG:32723')
            
            new_station_buffer = station_projected.geometry[0].buffer(self.catchment_radius)
            new_station_buffer = make_valid(new_station_buffer)
            
            zone_coverage = {}
            for idx, zone in zones_projected.iterrows():
                try:
                    zone_geom = make_valid(zone.geometry)
                    intersection = new_station_buffer.intersection(zone_geom)
                    
                    if not intersection.is_empty:
                        coverage_pct = (intersection.area / zone_geom.area) * 100
                        if coverage_pct > 0.1:
                            zone_id = str(idx + 1)
                            zone_coverage[zone_id] = round(coverage_pct, 2)
                except:
                    continue
            
            # Cache result
            if len(self.zone_cache) < 10000:
                self.zone_cache[cache_key] = zone_coverage
            
            return zone_coverage
            
        except:
            return {}
    
    def calculate_station_features(self, lat: float, lon: float, 
                                  existing_stations: List[Tuple[float, float]]) -> np.ndarray:
        """Calculate features with caching."""
        # Create cache key
        cache_key = (round(lat, 4), round(lon, 4), len(existing_stations))
        
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
        
        zone_coverages = self.get_station_zones_with_overlaps(lat, lon, existing_stations)
        weighted_features = np.zeros(len(self.feature_names))
        
        for zone_id, coverage_pct in zone_coverages.items():
            zone_int = int(zone_id)
            zone_row = self.features_df[self.features_df['zona'] == zone_int]
            
            if len(zone_row) > 0:
                zone_values = zone_row.iloc[0][self.feature_names].values
                weight = coverage_pct / 100.0
                weighted_features += zone_values * weight
        
        # Cache result
        if len(self.feature_cache) < 10000:
            self.feature_cache[cache_key] = weighted_features
        
        return weighted_features
    
    def evaluate_network_with_model(self, new_stations: List[Tuple[float, float]], 
                                   new_connections: List[Tuple[int, int]]) -> float:
        """Evaluate network with model."""
        try:
            all_features = [self.existing_features]
            
            for i, (lat, lon) in enumerate(new_stations):
                existing_for_overlap = self.existing_stations + new_stations[:i]
                features = self.calculate_station_features(lat, lon, existing_for_overlap)
                all_features.append(features.reshape(1, -1))
            
            if len(all_features) > 1:
                X = np.vstack(all_features)
            else:
                X = all_features[0]
            
            edges = []
            for idx1, idx2, _ in self.line_connections:
                edges.append([idx1, idx2])
                edges.append([idx2, idx1])
            
            for from_idx, to_idx in new_connections:
                edges.append([from_idx, to_idx])
                edges.append([to_idx, from_idx])
            
            edge_index = np.array(edges).T if edges else np.array([[], []])
            
            X_scaled = self.scaler.transform(X)
            x = torch.tensor(X_scaled, dtype=torch.float32)
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
            
            with torch.no_grad():
                predictions = self.model(x, edge_index_tensor)
            
            return predictions.sum().item()
            
        except Exception as e:
            return 0.0
    
    def generate_candidates_around_stations(self, reference_stations: List[Tuple[float, float]], 
                                           spacing: float = 0.5) -> List[Tuple[float, float]]:
        """Generate candidates with larger spacing for speed."""
        candidates = set()
        
        for ref_lat, ref_lon in reference_stations:
            for distance in np.arange(self.min_distance, self.max_distance + spacing, spacing):
                num_points = max(6, int(2 * np.pi * distance / spacing))
                
                for angle in np.linspace(0, 360, num_points, endpoint=False):
                    angle_rad = np.radians(angle)
                    
                    lat_offset = (distance / 111) * np.cos(angle_rad)
                    lon_offset = (distance / (111 * np.cos(np.radians(ref_lat)))) * np.sin(angle_rad)
                    
                    new_lat = ref_lat + lat_offset
                    new_lon = ref_lon + lon_offset
                    
                    candidates.add((round(new_lat, 6), round(new_lon, 6)))
        
        return list(candidates)
    
    def expand_network_beam_search(self, num_stations: int = 5, beam_width: int = 5):
        """
        Optimized beam search with parallel candidate evaluation.
        """
        print(f"\n{'='*70}")
        print(f"OPTIMIZED BEAM SEARCH EXPANSION")
        print(f"  Stations to add: {num_stations}")
        print(f"  Beam width: {beam_width}")
        print(f"  Parallel processing: {self.use_parallel}")
        print(f"{'='*70}")
        
        start_time = time.time()
        
        # Calculate baseline
        self.baseline_ridership = self.evaluate_network_with_model([], [])
        print(f"\n📊 Baseline ridership: {self.baseline_ridership:,.0f}")
        
        beam = [([], [], self.baseline_ridership)]
        
        # Add stations iteratively
        for station_num in range(1, num_stations + 1):
            print(f"\n📍 Adding station {station_num}/{num_stations}...")
            
            # Clear some cache if it's getting too large
            if len(self.zone_cache) > 5000:
                self.zone_cache.clear()
                print("  Cache cleared to maintain performance")
            
            next_beam = []
            
            if self.use_parallel and len(beam) > 1:
                # Parallel evaluation for multiple beam configurations
                with Pool(processes=min(4, cpu_count())) as pool:
                    all_results = []
                    
                    for beam_idx, (current_stations, current_connections, current_ridership) in enumerate(beam):
                        all_current_stations = self.existing_stations + current_stations
                        candidates = self.generate_candidates_around_stations(all_current_stations, spacing=0.5)
                        
                        # Filter valid candidates
                        valid_candidates = []
                        for lat, lon in candidates:
                            too_close = False
                            for ex_lat, ex_lon in all_current_stations:
                                if self.calculate_distance(lat, lon, ex_lat, ex_lon) < self.min_distance:
                                    too_close = True
                                    break
                            if not too_close:
                                valid_candidates.append((lat, lon))
                        
                        # Limit candidates
                        if len(valid_candidates) > 50:
                            np.random.seed(42 + station_num + beam_idx)
                            sample_indices = np.random.choice(len(valid_candidates), size=50, replace=False)
                            valid_candidates = [valid_candidates[i] for i in sample_indices]
                        
                        print(f"  Config {beam_idx+1}: {len(valid_candidates)} candidates")
                        
                        # Split candidates for parallel processing
                        chunk_size = max(10, len(valid_candidates) // 4)
                        candidate_chunks = [valid_candidates[i:i+chunk_size] 
                                          for i in range(0, len(valid_candidates), chunk_size)]
                        
                        # Prepare data for parallel processing
                        batch_data = [
                            (chunk, current_stations, current_connections,
                             self.existing_stations, self.min_distance, self.max_distance,
                             MODEL_PATH, SCALER_PATH, self.features_df, self.feature_names,
                             self.existing_features, self.line_connections, self.zones_gdf,
                             self.catchment_radius, self.existing_station_zones)
                            for chunk in candidate_chunks
                        ]
                        
                        # Process in parallel
                        chunk_results = pool.map(evaluate_candidate_batch, batch_data)
                        for results in chunk_results:
                            all_results.extend(results)
                    
                    next_beam = all_results
            
            else:
                # Sequential evaluation (fallback or single beam)
                for beam_idx, (current_stations, current_connections, current_ridership) in enumerate(beam):
                    all_current_stations = self.existing_stations + current_stations
                    candidates = self.generate_candidates_around_stations(all_current_stations, spacing=0.5)
                    
                    valid_candidates = []
                    for lat, lon in candidates:
                        too_close = False
                        for ex_lat, ex_lon in all_current_stations:
                            if self.calculate_distance(lat, lon, ex_lat, ex_lon) < self.min_distance:
                                too_close = True
                                break
                        if not too_close:
                            valid_candidates.append((lat, lon))
                    
                    if len(valid_candidates) > 50:
                        np.random.seed(42 + station_num + beam_idx)
                        sample_indices = np.random.choice(len(valid_candidates), size=50, replace=False)
                        valid_candidates = [valid_candidates[i] for i in sample_indices]
                    
                    print(f"  Config {beam_idx+1}: Testing {len(valid_candidates)} candidates...")
                    
                    for lat, lon in valid_candidates:
                        new_idx = len(all_current_stations)
                        
                        valid_connections = []
                        for idx, (station_lat, station_lon) in enumerate(all_current_stations):
                            dist = self.calculate_distance(lat, lon, station_lat, station_lon)
                            if self.min_distance <= dist <= self.max_distance:
                                valid_connections.append((idx, dist))
                        
                        if not valid_connections:
                            continue
                        
                        valid_connections.sort(key=lambda x: x[1])
                        for conn_idx, _ in valid_connections[:2]:
                            new_stations = current_stations + [(lat, lon)]
                            new_connections = current_connections + [(conn_idx, new_idx)]
                            
                            ridership = self.evaluate_network_with_model(new_stations, new_connections)
                            
                            if ridership > 0:
                                next_beam.append((new_stations, new_connections, ridership))
            
            if not next_beam:
                print("  ⚠️ No valid configurations found!")
                break
            
            # Keep top configurations
            next_beam.sort(key=lambda x: x[2], reverse=True)
            beam = next_beam[:beam_width]
            
            print(f"  Evaluated {len(next_beam)} configurations")
            print(f"  Best ridership: {beam[0][2]:,.0f} (+{beam[0][2] - self.baseline_ridership:,.0f})")
            if len(beam) > 1:
                print(f"  Beam range: {beam[-1][2]:,.0f} to {beam[0][2]:,.0f}")
            
            # Print cache statistics
            print(f"  Cache sizes - Zones: {len(self.zone_cache)}, "
                  f"Distances: {len(self.distance_cache)}, Features: {len(self.feature_cache)}")
        
        # Select best configuration
        if beam:
            best_stations, best_connections, best_ridership = beam[0]
            self.added_stations = best_stations
            self.added_connections = best_connections
            self.final_ridership = best_ridership
        else:
            self.added_stations = []
            self.added_connections = []
            self.final_ridership = self.baseline_ridership
        
        self.total_expansion_time = time.time() - start_time
        
        # Print summary
        print(f"\n{'='*70}")
        print("EXPANSION COMPLETE")
        print(f"{'='*70}")
        print(f"Added stations: {len(self.added_stations)}")
        print(f"Total gain: +{self.final_ridership - self.baseline_ridership:,.0f} trips")
        print(f"Percentage gain: +{(self.final_ridership/self.baseline_ridership - 1)*100:.1f}%")
        print(f"Time taken: {self.total_expansion_time:.1f} seconds")
        print(f"Final cache usage: {len(self.zone_cache) + len(self.distance_cache) + len(self.feature_cache)} entries")
        
        if self.added_stations:
            print("\nAdded stations:")
            all_stations = self.existing_stations + self.added_stations
            for i, ((lat, lon), (conn_from, conn_to)) in enumerate(zip(self.added_stations, self.added_connections)):
                from_station = all_stations[conn_from]
                dist = self.calculate_distance(lat, lon, from_station[0], from_station[1])
                conn_type = "existing" if conn_from < len(self.existing_stations) else f"new {conn_from - len(self.existing_stations) + 1}"
                print(f"  {i+1}. ({lat:.4f}, {lon:.4f}) - connected to {conn_type} station, {dist:.2f}km away")
    
    def visualize_network(self, output_path: str = None):
        """Visualize the expanded network."""
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Plot existing connections
        for from_idx, to_idx, _ in self.line_connections:
            if from_idx < len(self.existing_stations) and to_idx < len(self.existing_stations):
                from_lat, from_lon = self.existing_stations[from_idx]
                to_lat, to_lon = self.existing_stations[to_idx]
                ax.plot([from_lon, to_lon], [from_lat, to_lat], 
                       color='lightgray', linewidth=2, alpha=0.6, zorder=1)
        
        # Plot existing stations
        ex_lats = [s[0] for s in self.existing_stations]
        ex_lons = [s[1] for s in self.existing_stations]
        ax.scatter(ex_lons, ex_lats, c='lightblue', s=50, alpha=0.8, 
                  edgecolor='darkblue', linewidth=0.5, zorder=2, 
                  label='Estações Existentes')
        
        # Plot new stations and connections
        if self.added_stations:
            add_lats = [s[0] for s in self.added_stations]
            add_lons = [s[1] for s in self.added_stations]
            
            norm = mcolors.Normalize(vmin=1, vmax=len(self.added_stations))
            cmap = plt.cm.get_cmap('viridis')
            colors = [cmap(norm(i+1)) for i in range(len(self.added_stations))]
            
            all_stations = self.existing_stations + self.added_stations
            for i, (from_idx, to_idx) in enumerate(self.added_connections):
                from_lat, from_lon = all_stations[from_idx]
                to_lat, to_lon = all_stations[to_idx]
                
                edge_color = colors[i]
                ax.plot([from_lon, to_lon], [from_lat, to_lat], 
                       color=edge_color, linewidth=2.5, alpha=0.8, zorder=3)
            
            for i, (lat, lon, color) in enumerate(zip(add_lats, add_lons, colors)):
                ax.scatter(lon, lat, c=[color], s=200, zorder=4, 
                          edgecolor='black', linewidth=1.5)
                ax.annotate(str(i+1), xy=(lon, lat), 
                           xytext=(0, 0), textcoords='offset points',
                           fontsize=10, fontweight='bold', color='white',
                           ha='center', va='center', zorder=5)
        
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        
        gain = self.final_ridership - self.baseline_ridership if self.final_ridership else 0
        percent_gain = ((self.final_ridership/self.baseline_ridership - 1)*100) if self.baseline_ridership else 0
        
        title = f'Optimized GraphSAGE Metro Expansion\n'
        title += f'New Stations: {len(self.added_stations)} | '
        title += f'Ridership Gain: +{gain:,.0f} (+{percent_gain:.1f}%)\n'
        title += f'Algorithm: Parallel Beam Search | Time: {self.total_expansion_time:.1f}s'
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', framealpha=0.9, fontsize=10)
        
        if self.added_stations:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, label='Order of Addition', 
                               orientation='vertical', pad=0.01)
            cbar.set_ticks(range(1, len(self.added_stations)+1))
        
        stats_text = f'Network Statistics:\n'
        stats_text += f'Existing Stations: {len(self.existing_stations)}\n'
        stats_text += f'New Stations: {len(self.added_stations)}\n'
        stats_text += f'Baseline: {self.baseline_ridership:,.0f}\n' if self.baseline_ridership else ''
        stats_text += f'Final: {self.final_ridership:,.0f}\n' if self.final_ridership else ''
        stats_text += f'Time: {self.total_expansion_time:.1f}s\n'
        stats_text += f'Parallel: {self.use_parallel}'
        
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(OUTPUT_DIR, f'optimized_expansion_{timestamp}.png')
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Visualization saved to: {output_path}")
        plt.show()
        
        return fig, ax
    def export_results_to_csv(self, output_path: str = None):
        """Export detailed information about new stations to CSV."""
        if not self.added_stations:
            print("No stations to export")
            return
        
        # Prepare data for CSV
        station_data = []
        all_stations = self.existing_stations + self.added_stations
        
        for i, ((lat, lon), (conn_from, conn_to)) in enumerate(zip(self.added_stations, self.added_connections)):
            # Get connection details
            from_station = all_stations[conn_from]
            distance = self.calculate_distance(lat, lon, from_station[0], from_station[1])
            
            # Determine if connected to existing or new station
            if conn_from < len(self.existing_stations):
                connection_type = "existing"
                connected_station_id = self.station_ids[conn_from]
                connected_station_number = None
            else:
                connection_type = "new"
                connected_station_id = None
                connected_station_number = conn_from - len(self.existing_stations) + 1
            
            # Calculate coverage for this station
            zone_coverage = self.get_station_zones_with_overlaps(
                lat, lon, 
                self.existing_stations + self.added_stations[:i]
            )
            num_zones_covered = len(zone_coverage)
            total_coverage_pct = sum(zone_coverage.values())
            
            # Calculate feature values for this station
            features = self.calculate_station_features(
                lat, lon,
                self.existing_stations + self.added_stations[:i]
            )
            
            station_info = {
                'station_number': i + 1,
                'latitude': lat,
                'longitude': lon,
                'connected_to_type': connection_type,
                'connected_to_existing_id': connected_station_id,
                'connected_to_new_number': connected_station_number,
                'connection_distance_km': round(distance, 3),
                'zones_covered': num_zones_covered,
                'total_coverage_percent': round(total_coverage_pct, 2),
                'order_added': i + 1,
                'connection_from_index': conn_from,
                'connection_to_index': conn_to
            }
            
            # Add feature values
            for j, feature_name in enumerate(self.feature_names):
                # Clean feature name for column header
                clean_name = feature_name.replace(' ', '_').replace('–', '').replace('-', '')[:30]
                station_info[f'feature_{clean_name}'] = round(features[j], 2)
            
            station_data.append(station_info)
        
        # Create DataFrame
        df = pd.DataFrame(station_data)
        
        # Add summary row
        summary = {
            'station_number': 'SUMMARY',
            'latitude': None,
            'longitude': None,
            'connected_to_type': 'Total Gain:',
            'connected_to_existing_id': self.final_ridership - self.baseline_ridership,
            'connected_to_new_number': 'Pct Gain:',
            'connection_distance_km': round((self.final_ridership/self.baseline_ridership - 1)*100, 2),
            'zones_covered': 'Baseline:',
            'total_coverage_percent': self.baseline_ridership,
            'order_added': 'Final:',
            'connection_from_index': self.final_ridership
        }
        df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
        
        # Save to CSV
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(OUTPUT_DIR, f'new_stations_{timestamp}.csv')
        
        df.to_csv(output_path, index=False)
        print(f"\n✅ Station details exported to: {output_path}")
        
        # Print summary
        print("\nNew Stations Summary:")
        print(df[['station_number', 'latitude', 'longitude', 'connected_to_type', 
                'connection_distance_km', 'zones_covered']].to_string(index=False))
        
        return df


def main():
    """Main execution function."""
    # Configuration
    MIN_DISTANCE = 1.0  # km
    MAX_DISTANCE = 2.1  # km
    NUM_STATIONS = 50
    BEAM_WIDTH = 5  # Reduced for speed
    USE_PARALLEL = True  # Enable parallel processing
    
    print("="*70)
    print("OPTIMIZED CONFIGURATION")
    print("="*70)
    print(f"Min distance: {MIN_DISTANCE} km")
    print(f"Max distance: {MAX_DISTANCE} km")
    print(f"Stations to add: {NUM_STATIONS}")
    print(f"Beam width: {BEAM_WIDTH}")
    print(f"Parallel processing: {USE_PARALLEL}")
    
    # Create expander
    expander = GraphSAGENetworkExpander(
        min_distance=MIN_DISTANCE,
        max_distance=MAX_DISTANCE,
        use_parallel=USE_PARALLEL
    )
    
    # Expand network
    expander.expand_network_beam_search(
        num_stations=NUM_STATIONS,
        beam_width=BEAM_WIDTH
    )
    
    # Visualize results
    expander.visualize_network()

     #EXPORT TO CSV - ADD THIS
    expander.export_results_to_csv()
    
    # Save results
    results = {
        'algorithm': 'optimized_beam_search',
        'beam_width': BEAM_WIDTH,
        'min_distance': MIN_DISTANCE,
        'max_distance': MAX_DISTANCE,
        'parallel': USE_PARALLEL,
        'added_stations': expander.added_stations,
        'connections': expander.added_connections,
        'baseline_ridership': expander.baseline_ridership,
        'final_ridership': expander.final_ridership,
        'gain': expander.final_ridership - expander.baseline_ridership if expander.final_ridership else 0,
        'percentage_gain': ((expander.final_ridership/expander.baseline_ridership - 1)*100) if expander.baseline_ridership else 0,
        'time': expander.total_expansion_time
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(OUTPUT_DIR, f'optimized_results_{timestamp}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✅ Results saved to: {results_path}")
    
    return results

if __name__ == "__main__":
    results = main()