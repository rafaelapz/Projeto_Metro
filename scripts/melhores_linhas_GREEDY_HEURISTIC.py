

import sys
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely.errors import TopologicalError
import json
from typing import List, Tuple, Dict, Optional
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add path to import the zone coverage functions and dados.linhas
sys.path.append('/Users/ellazyngier/Documents/github/tccII/scripts/')
sys.path.append('/Users/ellazyngier/Documents/github/tccII/scripts/dados/')
from zone_coverage_function_for_new_nodes import NoOverlapLineFinder

# Import line definitions
from linhas import (
    azul1, verde2, vermelha3, amarela4, lilas5, diamante8,
    esmeralda9, turquesa10, coral11, safira12, jade13, prata15, rubi7,
    coral11_expresso_leste, expresso_aeroporto,
    current_and_express_lines
)

NUM_NEW_STATIONS_TO_ADD = 20 # Number of stations to add
CATCHMENT_RADIUS = 882  # Meters
MIN_STATION_SPACING = 1.6  # Minimum km between stations
MAX_STATION_SPACING = 3.0  # Maximum km between stations  
IDEAL_STATION_SPACING = 1.8  # Target spacing

# File paths
TRAVEL_MATRIX_PATH = '/Users/ellazyngier/Documents/github/tccII/scripts/dados/travel_matrix_VIAGENS_MOTORIZADAS_SOMENTE.npy'
STATIONS_CSV_PATH = '/Users/ellazyngier/Documents/github/tccII/scripts/dados/estacoes.csv'
ZONES_SHAPEFILE_PATH = '/Users/ellazyngier/Documents/github/tccII/Site_190225/002_Site Metro Mapas_190225/Shape/Zonas_2023.shp'
STATION_ZONES_JSON_PATH = '/Users/ellazyngier/Documents/github/tccII/scripts/dados/station_zones_882.json'
OUTPUT_DIR = '/Users/ellazyngier/Documents/github/tccII/scripts/dados/output_github'

class FilteredIncrementalExpander(NoOverlapLineFinder):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_station_distance = MIN_STATION_SPACING
        self.max_station_distance = MAX_STATION_SPACING
        self.ideal_station_distance = IDEAL_STATION_SPACING
        
        # Track incremental additions
        self.added_stations = []
        self.added_connections = []
        self.station_times = []  # Track time to add each station
        
        # Cache for zone calculations
        self._zone_cache = {}
        
        # Pre-project zones for efficiency
        self.zones_projected = self.zones_gdf.to_crs('EPSG:32723')
        self.zones_projected['geometry'] = self.zones_projected.geometry.buffer(0)
        
        # Filter stations to only those in current_and_express_lines
        self._filter_stations()
        
        # Create line connections for visualization
        self._create_line_connections()
        
        print("\n" + "="*60)
        print("OTIMIZADOR DE REDE DE METRÔ - CONFIGURAÇÃO")
        print("="*60)
        print(f"Configuração:")
        print(f"  Espaçamento mínimo: {self.min_station_distance} km")
        print(f"  Espaçamento ideal: {self.ideal_station_distance} km") 
        print(f"  Espaçamento máximo: {self.max_station_distance} km")
        print(f"  Estações consideradas: {len(self.filtered_stations)} de {len(self.existing_stations)}")
    
    def _filter_stations(self):
        """Filter stations to only those in current and express lines"""
        # Get all station IDs in current and express lines
        valid_station_ids = set()
        for line in current_and_express_lines:
            valid_station_ids.update(line)
        
        # Load station data from CSV with proper columns
        stations_df = pd.read_csv(STATIONS_CSV_PATH)
        
        # Check what columns we have
        print(f"CSV columns found: {list(stations_df.columns)}")
        
        # Filter existing stations based on codigo_estacao
        self.filtered_stations = []
        self.filtered_station_zones = {}
        self.filtered_station_names = []
        self.filtered_station_ids = []
        
        # Create a mapping from index to codigo_estacao for the original stations
        # The original stations were loaded in order from the CSV
        for idx in range(len(stations_df)):
            try:
                codigo_estacao = stations_df.iloc[idx]['codigo_estacao']
                
                # Check if this station is in our valid list
                if codigo_estacao in valid_station_ids:
                    # Get station info
                    lat = stations_df.iloc[idx]['latitude']
                    lon = stations_df.iloc[idx]['longitude']
                    nome = stations_df.iloc[idx]['nome']
                    
                    # Add to filtered lists
                    self.filtered_stations.append((lat, lon))
                    self.filtered_station_names.append(nome)
                    self.filtered_station_ids.append(codigo_estacao)
                    
                    # Get station zones if available
                    if str(codigo_estacao) in self.existing_station_zones:
                        new_idx = len(self.filtered_stations)
                        self.filtered_station_zones[str(new_idx)] = self.existing_station_zones[str(codigo_estacao)]
            except (KeyError, IndexError) as e:
                continue
        
        # Replace existing stations with filtered ones
        self.existing_stations = self.filtered_stations
        self.existing_station_zones = self.filtered_station_zones
        
        print(f"\nEstações filtradas: {len(self.filtered_stations)} estações nas linhas atuais e expressas")
        if self.filtered_station_ids:
            print(f"IDs das estações: min={min(self.filtered_station_ids)}, max={max(self.filtered_station_ids)}")
    
    def _create_line_connections(self):
        """Create connections between stations based on line definitions"""
        self.line_connections = []
        self.line_colors = {}
        
        # Define colors for each line
        line_configs = [
            (azul1, 'Azul', 'blue'),
            (verde2, 'Verde', 'green'),
            (vermelha3, 'Vermelha', 'red'),
            (amarela4, 'Amarela', 'gold'),
            (lilas5, 'Lilás', 'purple'),
            (diamante8, 'Diamante', 'gray'),
            (esmeralda9, 'Esmeralda', 'darkgreen'),
            (turquesa10, 'Turquesa', 'turquoise'),
            (coral11, 'Coral', 'coral'),
            (safira12, 'Safira', 'navy'),
            (jade13, 'Jade', 'mediumseagreen'),
            (prata15, 'Prata', 'silver'),
            (rubi7, 'Rubi', 'darkred'),
            (coral11_expresso_leste, 'Coral Expresso', 'orange'),
            (expresso_aeroporto, 'Expresso Aeroporto', 'orange')
        ]
        
        # Create mapping from station ID to index in filtered list
        station_id_to_index = {}
        if hasattr(self, 'filtered_station_ids'):
            for idx, station_id in enumerate(self.filtered_station_ids):
                station_id_to_index[station_id] = idx
        
        # Create connections for each line
        for line, name, color in line_configs:
            self.line_colors[name] = color
            for i in range(len(line) - 1):
                # Check if both stations in the connection exist in our filtered set
                if line[i] in station_id_to_index and line[i+1] in station_id_to_index:
                    self.line_connections.append((
                        station_id_to_index[line[i]], 
                        station_id_to_index[line[i+1]], 
                        color
                    ))
    
    def get_station_zones(self, lat: float, lon: float) -> Dict[str, float]:
        """Get station zones with error handling"""
        try:
            station_point = Point(lon, lat)
            station_gdf = gpd.GeoDataFrame([{'id': 'NEW'}], 
                                           geometry=[station_point],
                                           crs='EPSG:4326')
            
            station_projected = station_gdf.to_crs('EPSG:32723')
            station_buffer = station_projected.geometry[0].buffer(self.catchment_radius)
            station_buffer = station_buffer.simplify(1.0)
            
            zone_coverage = {}
            for idx, zone in self.zones_projected.iterrows():
                try:
                    zone_geom = zone.geometry
                    
                    if not zone_geom.is_valid:
                        zone_geom = zone_geom.buffer(0)
                    
                    try:
                        intersection = station_buffer.intersection(zone_geom)
                    except TopologicalError:
                        station_buffer_simple = station_buffer.simplify(10.0)
                        zone_geom_simple = zone_geom.simplify(10.0)
                        try:
                            intersection = station_buffer_simple.intersection(zone_geom_simple)
                        except:
                            continue
                    
                    if not intersection.is_empty:
                        try:
                            coverage_pct = (intersection.area / zone_geom.area) * 100
                            if coverage_pct > 0.1:
                                zone_id = str(idx + 1)
                                zone_coverage[zone_id] = round(coverage_pct, 2)
                        except:
                            continue
                            
                except Exception:
                    continue
            
            return zone_coverage
            
        except Exception as e:
            print(f"Aviso: Erro calculando zonas para estação em ({lat}, {lon}): {e}")
            return {}
    
    def get_station_zones_cached(self, lat: float, lon: float) -> Dict[str, float]:
        """Get station zones with caching"""
        cache_key = (round(lat, 4), round(lon, 4))
        if cache_key in self._zone_cache:
            return self._zone_cache[cache_key]
        
        zones = self.get_station_zones(lat, lon)
        self._zone_cache[cache_key] = zones
        return zones
    
    def calculate_station_pair_ridership(self, 
                                        station_a_zones: Dict[str, float],
                                        station_b_zones: Dict[str, float]) -> float:
        """Calculate ridership between two stations"""
        total_ridership = 0
        
        # A to B trips
        for zone_a, coverage_a in station_a_zones.items():
            zone_a_idx = int(zone_a) - 1
            if 0 <= zone_a_idx < 527:
                for zone_b, coverage_b in station_b_zones.items():
                    zone_b_idx = int(zone_b) - 1
                    if 0 <= zone_b_idx < 527:
                        trips_a_to_b = self.travel_matrix[zone_a_idx, zone_b_idx]
                        ridership_contrib = (coverage_a/100) * (coverage_b/100) * trips_a_to_b
                        total_ridership += ridership_contrib
        
        # B to A trips
        for zone_b, coverage_b in station_b_zones.items():
            zone_b_idx = int(zone_b) - 1
            if 0 <= zone_b_idx < 527:
                for zone_a, coverage_a in station_a_zones.items():
                    zone_a_idx = int(zone_a) - 1
                    if 0 <= zone_a_idx < 527:
                        trips_b_to_a = self.travel_matrix[zone_b_idx, zone_a_idx]
                        ridership_contrib = (coverage_b/100) * (coverage_a/100) * trips_b_to_a
                        total_ridership += ridership_contrib
        
        return total_ridership / 2
    
    def calculate_correct_network_ridership(self, verbose: bool = False) -> float:
        """Calculate total network ridership"""
        all_stations = {}
        
        # Existing (filtered) stations
        for station_id, zones in self.existing_station_zones.items():
            all_stations[f'EX_{station_id}'] = zones
        
        # Added stations
        for i, (lat, lon) in enumerate(self.added_stations):
            zones = self.get_station_zones_cached(lat, lon)
            if zones:
                all_stations[f'NEW_{i}'] = zones
        
        if verbose:
            print(f"\nCalculando demanda para {len(all_stations)} estações...")
        
        # Calculate ridership between all station pairs
        total_ridership = 0
        station_ids = list(all_stations.keys())
        
        for i in range(len(station_ids)):
            for j in range(i+1, len(station_ids)):
                station_a_zones = all_stations[station_ids[i]]
                station_b_zones = all_stations[station_ids[j]]
                
                pair_ridership = self.calculate_station_pair_ridership(
                    station_a_zones, station_b_zones
                )
                total_ridership += pair_ridership
        
        # Within-station trips
        for station_id, zones in all_stations.items():
            within_ridership = self.calculate_station_pair_ridership(zones, zones)
            total_ridership += within_ridership
        
        if verbose:
            print(f"Demanda total da rede: {total_ridership:,.0f}")
        
        return total_ridership
    
    def generate_candidate_locations(self, from_lat: float, from_lon: float, 
                                    num_candidates: int = 20) -> List[Tuple[float, float]]:
        """Generate candidate locations around a station"""
        candidates = []
        
        distances = np.linspace(self.ideal_station_distance - 0.3, 
                               min(self.ideal_station_distance + 0.5, self.max_station_distance), 
                               3)
        angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
        
        all_stations = self.existing_stations + self.added_stations
        
        for dist in distances:
            for angle in angles:
                new_lat = from_lat + (dist/111) * np.cos(angle)
                new_lon = from_lon + (dist/111) * np.sin(angle) / np.cos(np.radians(from_lat))
                
                # Check minimum spacing
                valid = True
                for station_lat, station_lon in all_stations:
                    if self.calculate_distance(new_lat, new_lon, 
                                              station_lat, station_lon) < self.min_station_distance:
                        valid = False
                        break
                
                if valid:
                    candidates.append((new_lat, new_lon))
        
        return candidates[:num_candidates]
    
    def find_best_next_station(self) -> Optional[Tuple[Tuple[float, float], int, float]]:
        """Find the best next station to add - checking ALL stations"""
        print("\nBuscando melhor localização para próxima estação...")
        
        current_ridership = self.calculate_correct_network_ridership()
        print(f"Demanda atual da rede: {current_ridership:,.0f}")
        
        best_station = None
        best_connection = None
        best_gain = 0
        
        all_current_stations = self.existing_stations + self.added_stations
        total_evaluated = 0
        
        # Include ALL stations - no sampling!
        search_stations = []
        
        # Add ALL existing stations
        for i, station in enumerate(self.existing_stations):
            search_stations.append((i, station))
        
        # Add ALL newly added stations
        for i, station in enumerate(self.added_stations):
            search_stations.append((len(self.existing_stations) + i, station))
        
        print(f"    Considerando conexões de TODAS as {len(search_stations)} estações:")
        print(f"      {len(self.existing_stations)} existentes")
        print(f"      {len(self.added_stations)} novas")
        print(f"    Isso vai demorar mais, mas garante encontrar a melhor conexão!")
        
        for from_idx, (from_lat, from_lon) in search_stations:
            candidates = self.generate_candidate_locations(from_lat, from_lon)
            
            for candidate_lat, candidate_lon in candidates:
                self.added_stations.append((candidate_lat, candidate_lon))
                
                new_ridership = self.calculate_correct_network_ridership()
                gain = new_ridership - current_ridership
                
                if gain > best_gain:
                    best_gain = gain
                    best_station = (candidate_lat, candidate_lon)
                    best_connection = from_idx
                
                self.added_stations.pop()
                
                total_evaluated += 1
                if total_evaluated % 500 == 0:
                    print(f"    Avaliadas {total_evaluated} candidatas...")
        
        print(f"    Total de candidatas avaliadas: {total_evaluated}")
        
        if best_station:
            return best_station, best_connection, best_gain
        return None
    
    def expand_network(self, num_new_stations: int = None):
        """Incrementally expand the network with timing"""
        if num_new_stations is None:
            num_new_stations = NUM_NEW_STATIONS_TO_ADD
            
        print(f"\n{'='*60}")
        print(f"EXPANSÃO INCREMENTAL DA REDE DE METRÔ")
        print(f"Considerando apenas linhas atuais e expressas")
        print(f"Adicionando {num_new_stations} novas estações")
        print(f"{'='*60}")
        
        # Start timing
        total_start_time = time.time()
        
        # Calculate baseline
        print("\nCalculando demanda base...")
        baseline_ridership = self.calculate_correct_network_ridership(verbose=False)
        print(f"Demanda base: {baseline_ridership:,.0f}")
        
        # Store detailed results for CSV
        self.expansion_results = []
        
        # Add stations one by one
        for i in range(num_new_stations):
            print(f"\n--- Adicionando estação {i+1}/{num_new_stations} ---")
            
            station_start_time = time.time()
            result = self.find_best_next_station()
            station_time = time.time() - station_start_time
            self.station_times.append(station_time)
            
            if result:
                new_station, connection_idx, gain = result
                
                # Add the station
                self.added_stations.append(new_station)
                self.added_connections.append((connection_idx, 
                                              len(self.existing_stations) + len(self.added_stations) - 1))
                
                # Get connection info - now from_idx is the actual station index
                if connection_idx < len(self.existing_stations):
                    # Connecting from an existing station
                    if connection_idx < len(self.filtered_station_names):
                        from_name = self.filtered_station_names[connection_idx]
                    else:
                        from_name = f"Estação existente {connection_idx}"
                else:
                    # Connecting from a newly added station
                    new_station_idx = connection_idx - len(self.existing_stations) + 1
                    from_name = f"Nova estação {new_station_idx}"
                
                # Get zone coverage
                zones = self.get_station_zones_cached(new_station[0], new_station[1])
                
                # Calculate current total
                current_total = self.calculate_correct_network_ridership()
                
                # Store results
                self.expansion_results.append({
                    'Ordem_Adicao': i + 1,
                    'Latitude': new_station[0],
                    'Longitude': new_station[1],
                    'Conectada_A': from_name,
                    'Zonas_Cobertas': len(zones),
                    'Ganho_Demanda': gain,
                    'Demanda_Total': current_total,
                    'Tempo_Busca_Segundos': station_time,
                    'Tempo_Acumulado_Segundos': sum(self.station_times)
                })
                
                print(f"  Adicionada em: ({new_station[0]:.4f}, {new_station[1]:.4f})")
                print(f"  Conectada a: {from_name}")
                print(f"  Ganho de demanda: +{gain:,.0f}")
                print(f"  Demanda total: {current_total:,.0f}")
                print(f"  Tempo de busca: {station_time:.1f} segundos")
                print(f"  Cache: {len(self._zone_cache)} localizações")
            else:
                print("  Nenhuma localização válida encontrada!")
                break
        
        # Total timing
        total_time = time.time() - total_start_time
        
        # Final summary
        final_ridership = self.calculate_correct_network_ridership()
        print(f"\n{'='*60}")
        print(f"EXPANSÃO CONCLUÍDA")
        print(f"Estações adicionadas: {len(self.added_stations)}")
        print(f"Demanda final: {final_ridership:,.0f}")
        print(f"Ganho total: +{final_ridership - baseline_ridership:,.0f}")
        print(f"Aumento percentual: {((final_ridership/baseline_ridership - 1)*100):.1f}%")
        print(f"\nANÁLISE DE TEMPO:")
        print(f"  Tempo total: {total_time:.1f} segundos")
        print(f"  Tempo médio por estação: {np.mean(self.station_times):.1f} segundos")
        print(f"  Tempo mínimo: {min(self.station_times):.1f} segundos")
        print(f"  Tempo máximo: {max(self.station_times):.1f} segundos")
        print(f"{'='*60}")
        
        self.baseline_ridership = baseline_ridership
        self.final_ridership = final_ridership
        self.total_expansion_time = total_time
        
        # Save CSV results
        self._save_results_csv()
        
        return self.added_stations, self.added_connections
    
    def _save_results_csv(self):
        """Save detailed results to CSV"""
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        
        # Create DataFrame
        df = pd.DataFrame(self.expansion_results)
        
        # Add summary statistics at the end
        summary_data = {
            'Ordem_Adicao': 'RESUMO',
            'Latitude': '',
            'Longitude': '',
            'Conectada_A': '',
            'Zonas_Cobertas': '',
            'Ganho_Demanda': self.final_ridership - self.baseline_ridership,
            'Demanda_Total': self.final_ridership,
            'Tempo_Busca_Segundos': self.total_expansion_time,
            'Tempo_Acumulado_Segundos': self.total_expansion_time
        }
        
        df = pd.concat([df, pd.DataFrame([summary_data])], ignore_index=True)
        
        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"expansao_metro_{timestamp}.csv"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"\n✓ Resultados salvos em: {filepath}")
    
    def visualize_network(self, output_path: str = None):
        """Visualize the expanded network with Portuguese labels"""
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Plot line connections first (as background) - ALL IN SAME COLOR
        for from_idx, to_idx, color in self.line_connections:
            if from_idx < len(self.existing_stations) and to_idx < len(self.existing_stations):
                from_lat, from_lon = self.existing_stations[from_idx]
                to_lat, to_lon = self.existing_stations[to_idx]
                # Use light gray for all existing edges
                ax.plot([from_lon, to_lon], [from_lat, to_lat], 
                       color='lightgray', linewidth=2, alpha=0.6, zorder=1)
        
        # Plot existing stations
        ex_lats = [s[0] for s in self.existing_stations]
        ex_lons = [s[1] for s in self.existing_stations]
        ax.scatter(ex_lons, ex_lats, c='lightblue', s=50, alpha=0.8, 
                  edgecolor='darkblue', linewidth=0.5, zorder=2, 
                  label='Estações Existentes')
        
        # Plot added stations with gradient colors
        if self.added_stations:
            add_lats = [s[0] for s in self.added_stations]
            add_lons = [s[1] for s in self.added_stations]
            
            # Create color gradient for added stations
            norm = mcolors.Normalize(vmin=1, vmax=len(self.added_stations))
            cmap = plt.cm.get_cmap('viridis')
            colors = [cmap(norm(i+1)) for i in range(len(self.added_stations))]
            
            # Plot connections for new stations WITH MATCHING COLORS
            all_stations = self.existing_stations + self.added_stations
            for i, (from_idx, to_idx) in enumerate(self.added_connections):
                # from_idx is the actual index in all_stations
                # to_idx is the index of the new station
                
                if from_idx < len(self.existing_stations):
                    from_lat, from_lon = self.existing_stations[from_idx]
                else:
                    # Connection from a newly added station
                    new_idx = from_idx - len(self.existing_stations)
                    from_lat, from_lon = self.added_stations[new_idx]
                
                # to_idx should be the new station being connected
                to_lat, to_lon = all_stations[to_idx]
                
                # Use the same color as the new station (station i+1)
                edge_color = colors[i]
                
                # All connections use solid lines
                ax.plot([from_lon, to_lon], [from_lat, to_lat], 
                       color=edge_color, linewidth=2.5, alpha=0.8, zorder=3)
            
            # Plot added stations with gradient
            for i, (lat, lon, color) in enumerate(zip(add_lats, add_lons, colors)):
                ax.scatter(lon, lat, c=[color], s=200, zorder=4, 
                          edgecolor='black', linewidth=1.5)
                # Number each station
                ax.annotate(str(i+1), xy=(lon, lat), 
                           xytext=(0, 0), textcoords='offset points',
                           fontsize=10, fontweight='bold', color='white',
                           ha='center', va='center', zorder=5)
        
        # Labels and title
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        
        gain = self.final_ridership - self.baseline_ridership
        percent_gain = ((self.final_ridership/self.baseline_ridership - 1)*100)
        
        title = f'Expansão Incremental da Rede de Metrô\n'
        title += f'Novas Estações: {len(self.added_stations)} | '
        title += f'Ganho de Demanda: +{gain:,.0f} (+{percent_gain:.1f}%)\n'
        title += f'Cores indicam ordem de adição (1=primeiro, {len(self.added_stations)}=último)'
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add legend for existing stations only
        ax.legend(loc='upper right', framealpha=0.9, fontsize=10)
        
        # Add colorbar for station order
        if self.added_stations:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, label='Ordem de Adição', 
                               orientation='vertical', pad=0.01)
            cbar.set_ticks(range(1, len(self.added_stations)+1))
        
        # Statistics box
        stats_text = f'Estatísticas da Rede:\n'
        stats_text += f'Estações Existentes: {len(self.existing_stations)}\n'
        stats_text += f'Novas Estações: {len(self.added_stations)}\n'
        stats_text += f'Demanda Base: {self.baseline_ridership:,.0f}\n'
        stats_text += f'Demanda Final: {self.final_ridership:,.0f}\n'
        stats_text += f'Tempo Total: {self.total_expansion_time:.1f}s'
        
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save figure
        plt.tight_layout()
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(OUTPUT_DIR, f'rede_expandida_{timestamp}.png')
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualização salva em: {output_path}")
        plt.show()
        
        return fig, ax


# Main execution
if __name__ == "__main__":
    print("\n" + "="*70)
    print("INICIANDO OTIMIZAÇÃO DE EXPANSÃO DA REDE DE METRÔ")
    print("="*70)
    
    start_time = time.time()
    
    # Initialize the expander
    expander = FilteredIncrementalExpander(
        travel_matrix_path=TRAVEL_MATRIX_PATH,
        existing_stations_csv=STATIONS_CSV_PATH,
        zones_shapefile_path=ZONES_SHAPEFILE_PATH,
        station_zones_json=STATION_ZONES_JSON_PATH,
        catchment_radius=CATCHMENT_RADIUS
    )
    
    # Expand the network
    added_stations, connections = expander.expand_network(num_new_stations=NUM_NEW_STATIONS_TO_ADD)
    
    # Create visualization
    expander.visualize_network()
    
    # Print final details
    if added_stations:
        print("\n" + "="*60)
        print("DETALHES DAS NOVAS ESTAÇÕES:")
        print("="*60)
        for i, (lat, lon) in enumerate(added_stations):
            zones = expander.get_station_zones_cached(lat, lon)
            print(f"Estação {i+1}: Lat={lat:.6f}, Lon={lon:.6f}, Cobre {len(zones)} zonas")
    
    elapsed = time.time() - start_time
    print(f"\n" + "="*60)
    print(f"Tempo total de execução: {elapsed:.1f} segundos")
    print(f"Processo concluído com sucesso!")
    print("="*60)