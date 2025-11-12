
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
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
import warnings
from dataclasses import dataclass, field
from collections import defaultdict
import math
import random

# Suppress warnings
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


@dataclass
class MCTSNode:
    """Node in the MCTS tree representing a network state."""
    state_stations: List[Tuple[float, float]]  # Added stations
    state_connections: List[Tuple[int, int]]  # Added connections
    ridership: float  # Current ridership
    cached_features: np.ndarray  # Cached feature matrix for new stations
    
    parent: Optional['MCTSNode'] = None
    children: Dict[Tuple[float, float, int], 'MCTSNode'] = field(default_factory=dict)  # (lat, lon, conn_idx) -> child
    
    visits: int = 0
    total_reward: float = 0.0
    
    # Action that led to this node
    action_station: Optional[Tuple[float, float]] = None  # Station added
    action_connection: Optional[Tuple[int, int]] = None  # Connection made
    
    def ucb1(self, c: float = math.sqrt(2)) -> float:
        """Calculate UCB1 value for node selection."""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.total_reward / self.visits
        exploration = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration
    
    def is_fully_expanded(self, max_children: int = 10) -> bool:
        """Check if node has been fully expanded."""
        return len(self.children) >= max_children

class MCTSNetworkExpander:
    def __init__(self, min_distance: float = 1.0, max_distance: float = 2.1, 
                 catchment_radius: int = 882, 
                 exploration_constant: float = 1.4,
                 rollout_depth: int = 3,
                 simulations_per_move: int = 100):

        self.min_distance = min_distance
        self.max_distance = max_distance
        self.catchment_radius = catchment_radius
        self.exploration_constant = exploration_constant
        self.rollout_depth = rollout_depth
        self.simulations_per_move = simulations_per_move
        
        # Initialize caches
        self.zone_cache = {}
        self.distance_cache = {}
        self.feature_calculation_cache = {}
        self.ridership_cache = {}  # Cache ridership evaluations
        
        print("="*70)
        print("EXPANSÃO DE REDE COM BUSCA DE ÁRVORE MONTE CARLO")
        print("="*70)
        print(f"Configuração:")
        print(f"  Distância mínima: {min_distance} km")
        print(f"  Distância máxima: {max_distance} km")
        print(f"  Raio de captação: {catchment_radius} m")
        print(f"  Constante de exploração: {exploration_constant}")
        print(f"  Profundidade de simulação: {rollout_depth}")
        print(f"  Simulações por movimento: {simulations_per_move}")
        
        # Load model and scaler
        print("\n📊 Carregando modelo treinado...")
        self.model = GraphSAGEModel(num_features=8)
        self.model.load_state_dict(torch.load(MODEL_PATH))
        self.model.eval()
        print("✅ Modelo carregado")
        
        with open(SCALER_PATH, 'rb') as f:
            self.scaler = pickle.load(f)
        print("✅ Normalizador carregado")
        
        # Load data
        print("\n📊 Carregando dados da rede...")
        self._load_existing_network()
        self._load_zone_data()
        self._load_features()
        
        # Initialize tracking
        self.added_stations = []
        self.added_connections = []
        self.baseline_ridership = None
        self.final_ridership = None
        self.total_expansion_time = 0
        self.total_simulations = 0
        
        print(f"✅ Sistema MCTS inicializado")
        
    def _load_existing_network(self):
        """Load existing stations and network structure."""
        df = pd.read_csv(os.path.join(DADOS_DIR, 'estacoes.csv'))
        
        active_stations = set()
        for line in current_and_express_lines:
            active_stations.update(line)
        
        with open(os.path.join(DADOS_DIR, 'station_zones_882.json'), 'r') as f:
            station_zones = json.load(f)
        
        self.station_ids = sorted([int(s) for s in active_stations if str(s) in station_zones])
        
        # Store station names for lookup
        self.station_names = {}
        self.existing_stations = []
        for sid in self.station_ids:
            row = df[df['codigo_estacao'] == sid]
            if not row.empty:
                self.existing_stations.append((row.iloc[0]['latitude'], row.iloc[0]['longitude']))
                self.station_names[sid] = row.iloc[0]['nome']
        
        print(f"✅ Carregadas {len(self.existing_stations)} estações existentes")
        
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
        print(f"✅ Carregadas {len(self.zones_gdf)} zonas")
        
        with open(os.path.join(DADOS_DIR, 'station_zones_882.json'), 'r') as f:
            self.existing_station_zones = json.load(f)
    
    def _load_features(self):
        """Load the 8 features for zones."""
        features_path = os.path.join(DADOS_DIR, '8_features.xlsx')
        self.features_df = pd.read_excel(features_path)
        self.feature_names = [col for col in self.features_df.columns if col != 'zona']
        print(f"✅ Carregadas {len(self.feature_names)} características")
        
        # Calculate and cache existing station features
        self.existing_features = self._aggregate_zones_to_stations(self.station_ids)
        self.existing_features_scaled = self.scaler.transform(self.existing_features)
        print(f"✅ Características pré-computadas para estações existentes")
    
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
        cache_key = (round(lat1, 6), round(lon1, 6), round(lat2, 6), round(lon2, 6))
        
        if cache_key in self.distance_cache:
            return self.distance_cache[cache_key]
        
        R = 6371
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * \
            np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
        dist = R * 2 * np.arcsin(np.sqrt(a))
        
        if len(self.distance_cache) < 10000:
            self.distance_cache[cache_key] = dist
        
        return dist
    
    def get_station_zones(self, lat: float, lon: float) -> Dict[str, float]:
        """Calculate zone coverage with caching."""
        cache_key = (round(lat, 5), round(lon, 5))
        
        if cache_key in self.zone_cache:
            return self.zone_cache[cache_key]
        
        try:
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
            
            if len(self.zone_cache) < 10000:
                self.zone_cache[cache_key] = zone_coverage
            
            return zone_coverage
        except:
            return {}
    
    def calculate_station_features(self, lat: float, lon: float) -> np.ndarray:
        """Calculate features with caching."""
        cache_key = (round(lat, 5), round(lon, 5))
        
        if cache_key in self.feature_calculation_cache:
            return self.feature_calculation_cache[cache_key].copy()
        
        zone_coverages = self.get_station_zones(lat, lon)
        weighted_features = np.zeros(len(self.feature_names))
        
        for zone_id, coverage_pct in zone_coverages.items():
            zone_int = int(zone_id)
            zone_row = self.features_df[self.features_df['zona'] == zone_int]
            
            if len(zone_row) > 0:
                zone_values = zone_row.iloc[0][self.feature_names].values
                weight = coverage_pct / 100.0
                weighted_features += zone_values * weight
        
        if len(self.feature_calculation_cache) < 10000:
            self.feature_calculation_cache[cache_key] = weighted_features.copy()
        
        return weighted_features
    
    def evaluate_network(self, new_stations: List[Tuple[float, float]], 
                        new_connections: List[Tuple[int, int]],
                        cached_features: np.ndarray = None) -> Tuple[float, np.ndarray]:
        """Evaluate network with caching."""
        # Create state key for caching
        state_key = (tuple(new_stations), tuple(new_connections))
        if state_key in self.ridership_cache:
            return self.ridership_cache[state_key], cached_features
        
        try:
            if cached_features is not None and len(cached_features) == len(new_stations):
                all_features_scaled = np.vstack([self.existing_features_scaled, cached_features])
            else:
                new_features = []
                for lat, lon in new_stations:
                    features = self.calculate_station_features(lat, lon)
                    new_features.append(features)
                
                if new_features:
                    new_features = np.array(new_features)
                    cached_features = self.scaler.transform(new_features)
                    all_features_scaled = np.vstack([self.existing_features_scaled, cached_features])
                else:
                    all_features_scaled = self.existing_features_scaled
                    cached_features = np.array([])
            
            # Build edge index
            edges = []
            for idx1, idx2, _ in self.line_connections:
                edges.append([idx1, idx2])
                edges.append([idx2, idx1])
            
            for from_idx, to_idx in new_connections:
                edges.append([from_idx, to_idx])
                edges.append([to_idx, from_idx])
            
            edge_index = np.array(edges).T if edges else np.array([[], []])
            
            # Predict
            x = torch.tensor(all_features_scaled, dtype=torch.float32)
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
            
            with torch.no_grad():
                predictions = self.model(x, edge_index_tensor)
            
            ridership = predictions.sum().item()
            
            # Cache result
            if len(self.ridership_cache) < 5000:
                self.ridership_cache[state_key] = ridership
            
            return ridership, cached_features
            
        except Exception as e:
            return 0.0, cached_features
    
    def generate_candidate_actions(self, node: MCTSNode, max_candidates: int = 30) -> List[Tuple[float, float, int]]:
        """Generate candidate stations to add from current state."""
        all_stations = self.existing_stations + node.state_stations
        
        # Generate candidates around ALL stations for better coverage
        candidates = []
        
        # Sample reference stations from entire network
        # Use more samples from existing stations to encourage network extension
        n_samples = min(20, len(all_stations))
        if len(node.state_stations) > 0:
            # Mix of existing (70%) and new stations (30%) as references
            n_existing_samples = int(n_samples * 0.7)
            n_new_samples = n_samples - n_existing_samples
            
            existing_sample = random.sample(self.existing_stations, 
                                          min(n_existing_samples, len(self.existing_stations)))
            new_sample = random.sample(node.state_stations, 
                                     min(n_new_samples, len(node.state_stations)))
            reference_stations = existing_sample + new_sample
        else:
            # Start with existing stations only
            reference_stations = random.sample(self.existing_stations, 
                                             min(n_samples, len(self.existing_stations)))
        
        # Generate candidates around sampled reference stations
        for ref_lat, ref_lon in reference_stations:
            # Generate 2-3 candidates per reference station
            for _ in range(2):
                distance = np.random.uniform(self.min_distance, self.max_distance)
                angle = np.random.uniform(0, 360)
                angle_rad = np.radians(angle)
                
                lat_offset = (distance / 111) * np.cos(angle_rad)
                lon_offset = (distance / (111 * np.cos(np.radians(ref_lat)))) * np.sin(angle_rad)
                
                new_lat = ref_lat + lat_offset
                new_lon = ref_lon + lon_offset
                
                # Check minimum distance constraint
                valid = True
                for ex_lat, ex_lon in all_stations:
                    if self.calculate_distance(new_lat, new_lon, ex_lat, ex_lon) < self.min_distance:
                        valid = False
                        break
                
                if valid:
                    # Find ALL valid connections and choose best one
                    new_idx = len(all_stations)
                    valid_connections = []
                    
                    for idx, (station_lat, station_lon) in enumerate(all_stations):
                        dist = self.calculate_distance(new_lat, new_lon, station_lat, station_lon)
                        if self.min_distance <= dist <= self.max_distance:
                            valid_connections.append((idx, dist))
                    
                    if valid_connections:
                        # Sort by distance and take closest valid connection
                        valid_connections.sort(key=lambda x: x[1])
                        best_idx, _ = valid_connections[0]
                        candidates.append((new_lat, new_lon, best_idx))
        
        # Shuffle to avoid any ordering bias
        random.shuffle(candidates)
        return candidates[:max_candidates]
    
    def select(self, node: MCTSNode) -> MCTSNode:
        """Select best child using UCB1."""
        while node.children:
            if not node.is_fully_expanded():
                return node
            
            # Select best child by UCB1
            best_child = max(node.children.values(), 
                           key=lambda n: n.ucb1(self.exploration_constant))
            node = best_child
        
        return node
    
    def expand(self, node: MCTSNode) -> MCTSNode:
        """Expand node by adding a new child."""
        # Generate candidate actions
        candidates = self.generate_candidate_actions(node)
        
        # Filter out already explored actions
        unexplored = []
        for lat, lon, conn_idx in candidates:
            action_key = (round(lat, 5), round(lon, 5), conn_idx)
            if action_key not in node.children:
                unexplored.append((lat, lon, conn_idx))
        
        if not unexplored:
            return node
        
        # Select random unexplored action
        lat, lon, conn_idx = random.choice(unexplored)
        
        # Create new state
        new_stations = node.state_stations + [(lat, lon)]
        new_idx = len(self.existing_stations) + len(new_stations) - 1
        new_connections = node.state_connections + [(conn_idx, new_idx)]
        
        # Calculate features for new station
        new_feature = self.calculate_station_features(lat, lon)
        new_feature_scaled = self.scaler.transform(new_feature.reshape(1, -1))
        
        if len(node.cached_features) > 0:
            cached_features = np.vstack([node.cached_features, new_feature_scaled])
        else:
            cached_features = new_feature_scaled
        
        # Evaluate new state
        ridership, _ = self.evaluate_network(new_stations, new_connections, cached_features)
        
        # Create child node
        child = MCTSNode(
            state_stations=new_stations,
            state_connections=new_connections,
            ridership=ridership,
            cached_features=cached_features,
            parent=node,
            action_station=(lat, lon),
            action_connection=(conn_idx, new_idx)
        )
        
        # Add to children
        action_key = (round(lat, 5), round(lon, 5), conn_idx)
        node.children[action_key] = child
        
        return child
    
    def simulate(self, node: MCTSNode) -> float:
        """Perform rollout simulation to estimate value."""
        current_stations = node.state_stations.copy()
        current_connections = node.state_connections.copy()
        current_ridership = node.ridership
        
        # Perform random rollout
        for _ in range(min(self.rollout_depth, 50 - len(current_stations))):
            all_stations = self.existing_stations + current_stations
            
            # Generate random valid station - sample from ENTIRE network
            found_valid = False
            for _ in range(20):  # Try up to 20 times
                # Sample uniformly from all stations (no recency bias)
                ref_station = random.choice(all_stations)
                distance = np.random.uniform(self.min_distance, self.max_distance)
                angle = np.random.uniform(0, 360)
                angle_rad = np.radians(angle)
                
                lat_offset = (distance / 111) * np.cos(angle_rad)
                lon_offset = (distance / (111 * np.cos(np.radians(ref_station[0])))) * np.sin(angle_rad)
                
                new_lat = ref_station[0] + lat_offset
                new_lon = ref_station[1] + lon_offset
                
                # Check constraints
                valid = True
                for ex_lat, ex_lon in all_stations:
                    if self.calculate_distance(new_lat, new_lon, ex_lat, ex_lon) < self.min_distance:
                        valid = False
                        break
                
                if valid:
                    # Find best connection (closest valid station)
                    best_connection = None
                    best_distance = float('inf')
                    
                    for idx, (station_lat, station_lon) in enumerate(all_stations):
                        dist = self.calculate_distance(new_lat, new_lon, station_lat, station_lon)
                        if self.min_distance <= dist <= self.max_distance and dist < best_distance:
                            best_connection = idx
                            best_distance = dist
                    
                    if best_connection is not None:
                        current_stations.append((new_lat, new_lon))
                        new_idx = len(all_stations)
                        current_connections.append((best_connection, new_idx))
                        found_valid = True
                        break
            
            if not found_valid:
                break
        
        # Evaluate final state
        final_ridership, _ = self.evaluate_network(current_stations, current_connections)
        
        # Return improvement over baseline
        return final_ridership - self.baseline_ridership
    
    def backpropagate(self, node: MCTSNode, reward: float):
        """Backpropagate reward up the tree."""
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent
    
    def get_best_action(self, root: MCTSNode) -> Optional[MCTSNode]:
        """Get best action based on visit count."""
        if not root.children:
            return None
        
        # Choose child with most visits (most robust)
        return max(root.children.values(), key=lambda n: n.visits)
    
    def expand_network_mcts(self, num_stations: int = 5):
        """
        Expand network using MCTS.
        """
        print(f"\n{'='*70}")
        print(f"EXPANSÃO POR BUSCA DE ÁRVORE MONTE CARLO")
        print(f"  Estações a adicionar: {num_stations}")
        print(f"  Simulações por movimento: {self.simulations_per_move}")
        print(f"  Constante de exploração: {self.exploration_constant}")
        print(f"{'='*70}")
        
        start_time = time.time()
        
        # Calculate baseline
        self.baseline_ridership, _ = self.evaluate_network([], [])
        print(f"\n📊 Passageiros base: {self.baseline_ridership:,.0f}")
        
        # Initialize tracking
        self.incremental_ridership = []  # Track incremental gains
        previous_ridership = self.baseline_ridership
        
        # Initialize with root node
        current_node = MCTSNode(
            state_stations=[],
            state_connections=[],
            ridership=self.baseline_ridership,
            cached_features=np.array([])
        )
        
        # Build network iteratively
        for station_num in range(1, num_stations + 1):
            print(f"\n📍 Adicionando estação {station_num}/{num_stations}...")
            
            # Run MCTS simulations
            for sim in tqdm(range(self.simulations_per_move), desc="  Simulações", leave=False):
                # Selection
                leaf = self.select(current_node)
                
                # Expansion
                if leaf.visits > 0 and len(leaf.state_stations) < num_stations:
                    leaf = self.expand(leaf)
                
                # Simulation
                reward = self.simulate(leaf)
                
                # Backpropagation
                self.backpropagate(leaf, reward)
                
                self.total_simulations += 1
            
            # Select best action
            best_child = self.get_best_action(current_node)
            
            if best_child is None:
                print("  ⚠️ Nenhuma ação válida encontrada!")
                break
            
            # Move to best child
            current_node = best_child
            
            # Calculate incremental gain
            incremental_gain = current_node.ridership - previous_ridership
            self.incremental_ridership.append(incremental_gain)
            previous_ridership = current_node.ridership
            
            # Report progress
            gain = current_node.ridership - self.baseline_ridership
            print(f"  Estação adicionada em ({best_child.action_station[0]:.4f}, {best_child.action_station[1]:.4f})")
            print(f"  Passageiros atual: {current_node.ridership:,.0f} (+{gain:,.0f} total)")
            print(f"  Ganho incremental: +{incremental_gain:,.0f}")
            print(f"  Visitas a este nó: {current_node.visits}")
            
            # Clear caches periodically
            if len(self.feature_calculation_cache) > 5000:
                self.feature_calculation_cache.clear()
            if len(self.ridership_cache) > 5000:
                self.ridership_cache.clear()
        
        # Store final results
        self.added_stations = current_node.state_stations
        self.added_connections = current_node.state_connections
        self.final_ridership = current_node.ridership
        self.total_expansion_time = time.time() - start_time
        
        # Print summary
        print(f"\n{'='*70}")
        print("EXPANSÃO COMPLETA")
        print(f"{'='*70}")
        print(f"Estações adicionadas: {len(self.added_stations)}")
        print(f"Ganho total: +{self.final_ridership - self.baseline_ridership:,.0f} viagens")
        print(f"Ganho percentual: +{(self.final_ridership/self.baseline_ridership - 1)*100:.1f}%")
        print(f"Tempo total: {self.total_expansion_time:.1f} segundos")
        print(f"Total de simulações: {self.total_simulations}")
        print(f"Simulações/segundo: {self.total_simulations/self.total_expansion_time:.1f}")
        
        if self.added_stations:
            print("\nEstações adicionadas:")
            all_stations = self.existing_stations + self.added_stations
            for i, ((lat, lon), (conn_from, conn_to)) in enumerate(zip(self.added_stations, self.added_connections)):
                from_station = all_stations[conn_from]
                dist = self.calculate_distance(lat, lon, from_station[0], from_station[1])
                
                # Get station name if connecting to existing station
                if conn_from < len(self.existing_stations):
                    station_id = self.station_ids[conn_from]
                    station_name = self.station_names[station_id]
                    conn_type = f"estação existente '{station_name}'"
                else:
                    conn_type = f"nova estação {conn_from - len(self.existing_stations) + 1}"
                    
                print(f"  {i+1}. ({lat:.4f}, {lon:.4f}) - conectada a {conn_type}, {dist:.2f}km de distância")
    
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
            cmap = plt.cm.get_cmap('plasma')  # Different colormap for MCTS
            colors = [cmap(norm(i+1)) for i in range(len(self.added_stations))]
            
            all_stations = self.existing_stations + self.added_stations
            for i, (from_idx, to_idx) in enumerate(self.added_connections):
                from_lat, from_lon = all_stations[from_idx]
                to_lat, to_lon = all_stations[to_idx]
                
                edge_color = colors[i]
                ax.plot([from_lon, to_lon], [from_lat, to_lat], 
                       color=edge_color, linewidth=2.5, alpha=0.8, zorder=3)
            
            # Plot stations with smaller numbers
            for i, (lat, lon, color) in enumerate(zip(add_lats, add_lons, colors)):
                ax.scatter(lon, lat, c=[color], s=150, zorder=4,  # Reduced size from 200
                          edgecolor='black', linewidth=1.2)  # Thinner edge
                
                # Only show numbers for every 10th station if there are many stations
                if len(self.added_stations) > 50:
                    if (i+1) % 10 == 0 or i == 0 or i == len(self.added_stations) - 1:
                        ax.annotate(str(i+1), xy=(lon, lat), 
                                   xytext=(0, 0), textcoords='offset points',
                                   fontsize=7, fontweight='bold', color='white',  # Smaller font
                                   ha='center', va='center', zorder=5)
                else:
                    # Show all numbers for fewer stations with smaller font
                    ax.annotate(str(i+1), xy=(lon, lat), 
                               xytext=(0, 0), textcoords='offset points',
                               fontsize=8, fontweight='bold', color='white',  # Reduced from 10
                               ha='center', va='center', zorder=5)
        
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        
        gain = self.final_ridership - self.baseline_ridership if self.final_ridership else 0
        percent_gain = ((self.final_ridership/self.baseline_ridership - 1)*100) if self.baseline_ridership else 0
        
        title = f'Expansão do Metrô por Busca de Árvore Monte Carlo\n'
        title += f'Novas Estações: {len(self.added_stations)} | '
        title += f'Ganho de Passageiros: +{gain:,.0f} (+{percent_gain:.1f}%)\n'
        title += f'Simulações: {self.total_simulations} | Tempo: {self.total_expansion_time:.1f}s'
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', framealpha=0.9, fontsize=10)
        
        if self.added_stations:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, label='Ordem de Adição', 
                               orientation='vertical', pad=0.01)
            
            # Set colorbar ticks intelligently based on number of stations
            n_stations = len(self.added_stations)
            if n_stations <= 25:
                # Show every 5 stations for small numbers
                tick_interval = 5
            elif n_stations <= 50:
                # Show every 10 stations
                tick_interval = 10
            else:
                # Show every 25 stations for large numbers
                tick_interval = 25
            
            # Generate tick positions
            tick_positions = list(range(1, n_stations + 1, tick_interval))
            # Always include the last station
            if n_stations not in tick_positions:
                tick_positions.append(n_stations)
            
            cbar.set_ticks(tick_positions)
            cbar.set_ticklabels(tick_positions)
        
        stats_text = f'Estatísticas MCTS:\n'
        stats_text += f'Estações Existentes: {len(self.existing_stations)}\n'
        stats_text += f'Novas Estações: {len(self.added_stations)}\n'
        stats_text += f'Base: {self.baseline_ridership:,.0f}\n' if self.baseline_ridership else ''
        stats_text += f'Final: {self.final_ridership:,.0f}\n' if self.final_ridership else ''
        stats_text += f'Simulações: {self.total_simulations}\n'
        stats_text += f'Exploração: {self.exploration_constant}'
        
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(OUTPUT_DIR, f'mcts_expansion_{timestamp}.png')
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Visualização salva em: {output_path}")
        plt.show()
        
        return fig, ax
    
    def export_results_to_csv(self, output_path: str = None):
        """Export detailed information about new stations to CSV."""
        if not self.added_stations:
            print("Nenhuma estação para exportar")
            return
        
        station_data = []
        all_stations = self.existing_stations + self.added_stations
        
        for i, ((lat, lon), (conn_from, conn_to)) in enumerate(zip(self.added_stations, self.added_connections)):
            from_station = all_stations[conn_from]
            distance = self.calculate_distance(lat, lon, from_station[0], from_station[1])
            
            if conn_from < len(self.existing_stations):
                connection_type = "existing"
                connected_station_id = self.station_ids[conn_from]
                connected_station_number = None
            else:
                connection_type = "new"
                connected_station_id = None
                connected_station_number = conn_from - len(self.existing_stations) + 1
            
            zone_coverage = self.get_station_zones(lat, lon)
            num_zones_covered = len(zone_coverage)
            total_coverage_pct = sum(zone_coverage.values())
            
            features = self.calculate_station_features(lat, lon)
            
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
                'order_added': i + 1
            }
            
            for j, feature_name in enumerate(self.feature_names):
                clean_name = feature_name.replace(' ', '_').replace('–', '').replace('-', '')[:30]
                station_info[f'feature_{clean_name}'] = round(features[j], 2)
            
            station_data.append(station_info)
        
        df = pd.DataFrame(station_data)
        
        # Add summary
        summary = {
            'station_number': 'SUMMARY',
            'latitude': None,
            'longitude': None,
            'connected_to_type': 'Algorithm:',
            'connected_to_existing_id': 'MCTS',
            'connected_to_new_number': 'Total Gain:',
            'connection_distance_km': self.final_ridership - self.baseline_ridership,
            'zones_covered': 'Simulations:',
            'total_coverage_percent': self.total_simulations,
            'order_added': 'Time (s):',
        }
        df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(OUTPUT_DIR, f'mcts_stations_{timestamp}.csv')
        
        df.to_csv(output_path, index=False)
        print(f"\n✅ Detalhes das estações exportados para: {output_path}")
        
        return df
    
    def export_resultados_mcts(self):
        """Export results in the specific format requested to resultados_MCTS.csv"""
        if not self.added_stations:
            print("Nenhuma estação para exportar")
            return
        
        # Prepare data for the specific CSV format
        data_rows = []
        
        for i, ((lat, lon), (conn_from, conn_to)) in enumerate(zip(self.added_stations, self.added_connections)):
            # Get connected station name
            if conn_from < len(self.existing_stations):
                station_id = self.station_ids[conn_from]
                connected_station_name = self.station_names[station_id]
            else:
                connected_station_name = f"Nova Estação {conn_from - len(self.existing_stations)}"
            
            # Get incremental ridership gain
            incremental_gain = self.incremental_ridership[i] if i < len(self.incremental_ridership) else 0
            
            data_rows.append({
                'numero_estacao': i + 1,
                'latitude': round(lat, 6),
                'longitude': round(lon, 6),
                'conectada_a': connected_station_name,
                'viagens_adicionadas': round(incremental_gain, 0)
            })
        
        # Create DataFrame
        df = pd.DataFrame(data_rows)
        
        # Add total row
        total_row = {
            'numero_estacao': 'TOTAL',
            'latitude': '',
            'longitude': '',
            'conectada_a': 'Ganho Total',
            'viagens_adicionadas': round(self.final_ridership - self.baseline_ridership, 0)
        }
        df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
        
        # Save to specific path
        output_path = '/Users/ellazyngier/Documents/github/tccII/scripts/resultados/resultados_MCTS.csv'
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\n✅ Resultados MCTS salvos em: {output_path}")
        
        # Also print summary
        print("\nResumo das Estações Adicionadas:")
        print(df.to_string(index=False))
        
        return df


def main():
    """Main execution function."""
    # Configuration
    MIN_DISTANCE = 1.0  # km
    MAX_DISTANCE = 10  # km
    NUM_STATIONS = 100
    EXPLORATION_CONSTANT = 1.4  # UCB exploration parameter (sqrt(2) ≈ 1.414)
    ROLLOUT_DEPTH = 3  # Depth of random rollouts
    SIMULATIONS_PER_MOVE = 100  # Number of simulations per station addition
    
    print("="*70)
    print("CONFIGURAÇÃO MCTS")
    print("="*70)
    print(f"Distância mínima: {MIN_DISTANCE} km")
    print(f"Distância máxima: {MAX_DISTANCE} km")
    print(f"Estações a adicionar: {NUM_STATIONS}")
    print(f"Constante de exploração: {EXPLORATION_CONSTANT}")
    print(f"Profundidade de simulação: {ROLLOUT_DEPTH}")
    print(f"Simulações por movimento: {SIMULATIONS_PER_MOVE}")
    print(f"\nTotal esperado de simulações: ~{NUM_STATIONS * SIMULATIONS_PER_MOVE}")
    
    # Create expander
    expander = MCTSNetworkExpander(
        min_distance=MIN_DISTANCE,
        max_distance=MAX_DISTANCE,
        exploration_constant=EXPLORATION_CONSTANT,
        rollout_depth=ROLLOUT_DEPTH,
        simulations_per_move=SIMULATIONS_PER_MOVE
    )
    
    # Expand network
    expander.expand_network_mcts(num_stations=NUM_STATIONS)
    
    # Visualize results
    expander.visualize_network()
    
    # Export to specific CSV format
    expander.export_resultados_mcts()
    
    # Also export detailed results
    expander.export_results_to_csv()
    
    # Save results JSON
    results = {
        'algorithm': 'monte_carlo_tree_search',
        'exploration_constant': EXPLORATION_CONSTANT,
        'rollout_depth': ROLLOUT_DEPTH,
        'simulations_per_move': SIMULATIONS_PER_MOVE,
        'total_simulations': expander.total_simulations,
        'min_distance': MIN_DISTANCE,
        'max_distance': MAX_DISTANCE,
        'added_stations': expander.added_stations,
        'connections': expander.added_connections,
        'baseline_ridership': expander.baseline_ridership,
        'final_ridership': expander.final_ridership,
        'gain': expander.final_ridership - expander.baseline_ridership if expander.final_ridership else 0,
        'percentage_gain': ((expander.final_ridership/expander.baseline_ridership - 1)*100) if expander.baseline_ridership else 0,
        'time': expander.total_expansion_time,
        'simulations_per_second': expander.total_simulations / expander.total_expansion_time if expander.total_expansion_time > 0 else 0,
        'incremental_gains': expander.incremental_ridership
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(OUTPUT_DIR, f'mcts_results_{timestamp}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✅ Resultados JSON salvos em: {results_path}")
    
    return results

if __name__ == "__main__":
    results = main()