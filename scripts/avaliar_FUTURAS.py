
import pickle
import numpy as np
import pandas as pd
import json
import sys
import os
import warnings
import time
from typing import List, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
from geopy.distance import geodesic

warnings.filterwarnings('ignore')

sys.path.append('/Users/ellazyngier/Documents/github/tccII/scripts/dados')
sys.path.append('/Users/ellazyngier/Documents/github/tccII/scripts')

from dados.linhas import current_and_express_lines
from calculate_system_ridership import load_data as load_od_data

DADOS_DIR = '/Users/ellazyngier/Documents/github/tccII/scripts/dados'
OUTPUT_DIR = '/Users/ellazyngier/Documents/github/tccII/scripts/dados/futuras'
RESULTS_DIR = '/Users/ellazyngier/Documents/github/tccII/scripts/resultados/expansion'


HIDDEN_DIM = 64
DROPOUT = 0.2


os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


class GraphSAGEModel(nn.Module):
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


class IndividualLineVisualizer:
    def __init__(self):
        """Initialize the evaluator."""
        self.model = None
        self.scaler = None
        self.features_df = None
        self.feature_names = None
        self.station_zones = None
        self.travel_matrix = None
        self.existing_stations_df = None
        self.station_ids = []
        self.station_coordinates = {}
        
        print("📊 Loading evaluation system...")
        self._load_trained_model()
        self._load_features()
        self._load_existing_network()
        self._load_travel_matrix()
        self._load_all_coordinates()
    
    def _load_trained_model(self):
        print("   Loading model...")
        model_path = os.path.join(DADOS_DIR, 'graphsage_model.pt')
        self.model = GraphSAGEModel(num_features=8)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        scaler_path = os.path.join(DADOS_DIR, 'scaler.pkl')
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
    
    def _load_features(self):
        features_path = os.path.join(DADOS_DIR, '8_features.xlsx')
        self.features_df = pd.read_excel(features_path)
        self.feature_names = [col for col in self.features_df.columns if col != 'zona']
        
        coverage_path = os.path.join(DADOS_DIR, 'station_zones_882.json')
        with open(coverage_path, 'r') as f:
            self.station_zones = json.load(f)
    
    def _load_existing_network(self):
        self.existing_stations_df = pd.read_csv(os.path.join(DADOS_DIR, 'estacoes.csv'))
        
        active_stations = set()
        for line in current_and_express_lines:
            active_stations.update(line)
        
        self.station_ids = sorted([int(s) for s in active_stations if str(s) in self.station_zones])
    
    def _load_travel_matrix(self):
        _, self.travel_matrix = load_od_data()
    
    def _load_all_coordinates(self):
        for _, row in self.existing_stations_df.iterrows():
            sid = row['codigo_estacao']
            self.station_coordinates[sid] = (row['latitude'], row['longitude'])
    
    def _calculate_line_length_km(self, station_ids):
        """Calculate line length in km."""
        total_distance = 0
        for i in range(len(station_ids) - 1):
            if station_ids[i] in self.station_coordinates and station_ids[i+1] in self.station_coordinates:
                coord1 = self.station_coordinates[station_ids[i]]
                coord2 = self.station_coordinates[station_ids[i+1]]
                distance = geodesic(coord1, coord2).kilometers
                total_distance += distance
        return total_distance
    
    def _aggregate_features_for_stations(self, station_ids, station_zones_dict):
        station_features = []
        for station_id in station_ids:
            zone_coverages = station_zones_dict[str(station_id)]
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
    
    def _create_graph_edges(self, station_ids, include_new_line=False, new_line_stations=None):
        edges = []
        
        for line in current_and_express_lines:
            for i in range(len(line) - 1):
                if line[i] in station_ids and line[i+1] in station_ids:
                    idx1 = station_ids.index(line[i])
                    idx2 = station_ids.index(line[i+1])
                    edges.append([idx1, idx2])
                    edges.append([idx2, idx1])
        
        if include_new_line and new_line_stations:
            for i in range(len(new_line_stations) - 1):
                if new_line_stations[i] in station_ids and new_line_stations[i+1] in station_ids:
                    idx1 = station_ids.index(new_line_stations[i])
                    idx2 = station_ids.index(new_line_stations[i+1])
                    edges.append([idx1, idx2])
                    edges.append([idx2, idx1])
        
        edges = list(set(map(tuple, edges)))
        edge_index = np.array(edges).T if edges else np.array([[], []])
        return edge_index
    
    def _calculate_network_ridership(self, station_ids, station_zones_dict, 
                                    include_new_line=False, new_line_stations=None):
        X = self._aggregate_features_for_stations(station_ids, station_zones_dict)
        edge_index = self._create_graph_edges(station_ids, include_new_line, new_line_stations)
        X_scaled = self.scaler.transform(X)
        
        x = torch.tensor(X_scaled, dtype=torch.float32)
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
        
        with torch.no_grad():
            predictions = self.model(x, edge_index_tensor)
        
        ridership = predictions.numpy()
        return {
            'total': ridership.sum(),
            'mean': ridership.mean(),
            'by_station': {sid: ridership[i] for i, sid in enumerate(station_ids)}
        }
    
    def evaluate_and_visualize_line(self, station_ids, line_name, status="", file_name=None):
        """Evaluate a single line and create its individual visualization."""
        
        start_time = time.time()
        
        # Analysis
        new_stations = []
        connections = []
        
        for sid in station_ids:
            if sid in self.station_ids:
                connections.append(sid)
            else:
                new_stations.append(sid)
        
        # Calculate metrics
        eval_start = time.time()
        baseline = self._calculate_network_ridership(
            self.station_ids, self.station_zones, include_new_line=False
        )
        
        expanded_station_ids = self.station_ids.copy()
        expanded_station_zones = self.station_zones.copy()
        
        for sid in new_stations:
            if sid not in expanded_station_ids:
                expanded_station_ids.append(sid)
                expanded_station_zones[str(sid)] = self.station_zones.get(
                    str(sid), 
                    self.station_zones.get(str(connections[0]), {}) if connections else {}
                )
        
        expanded_station_ids = sorted(expanded_station_ids)
        
        expanded = self._calculate_network_ridership(
            expanded_station_ids, expanded_station_zones,
            include_new_line=True, new_line_stations=station_ids
        )
        eval_time = time.time() - eval_start
        
        # Calculate all metrics
        total_impact = expanded['total'] - baseline['total']
        percent_improvement = (total_impact / baseline['total']) * 100 if baseline['total'] > 0 else 0
        line_length_km = self._calculate_line_length_km(station_ids)
        impact_per_station = total_impact / len(station_ids) if len(station_ids) > 0 else 0
        impact_per_km = total_impact / line_length_km if line_length_km > 0 else 0
        
        # Create individual figure
        viz_start = time.time()
        fig = plt.figure(figsize=(14, 10))
        ax = plt.subplot(111)
        
        # Plot existing network
        existing_edges = []
        for line in current_and_express_lines:
            for i in range(len(line) - 1):
                if line[i] in self.station_coordinates and line[i+1] in self.station_coordinates:
                    existing_edges.append((line[i], line[i+1]))
        
        # Draw existing network in purple to match stations
        for edge in existing_edges:
            sid1, sid2 = edge
            if sid1 in self.station_coordinates and sid2 in self.station_coordinates:
                lat1, lon1 = self.station_coordinates[sid1]
                lat2, lon2 = self.station_coordinates[sid2]
                ax.plot([lon1, lon2], [lat1, lat2], 
                       color='#9370DB', linewidth=1.5, alpha=0.5, zorder=1)  # Purple color, thinner line
        
        # Define colors
        EXISTING_COLOR = '#8A2BE2'  
        EXISTING_EDGE = '#4B0082'    # Darker purple for edge
        NEW_LINE_COLOR = '#DC143C'   # Crimson red
        NEW_LINE_EDGE = '#8B0000'    # Dark red for edge
        
        # Plot existing stations with purple and matching edge
        for sid in self.station_ids:
            if sid in self.station_coordinates:
                lat, lon = self.station_coordinates[sid]
                ax.scatter(lon, lat, c=EXISTING_COLOR, s=50, alpha=0.8,
                          edgecolor=EXISTING_EDGE, linewidth=0.5, zorder=2)
        
        # Plot new line connections
        for i in range(len(station_ids) - 1):
            sid1, sid2 = station_ids[i], station_ids[i+1]
            if sid1 in self.station_coordinates and sid2 in self.station_coordinates:
                lat1, lon1 = self.station_coordinates[sid1]
                lat2, lon2 = self.station_coordinates[sid2]
                ax.plot([lon1, lon2], [lat1, lat2], 
                       color=NEW_LINE_COLOR, linewidth=2.5, alpha=0.8, zorder=3)

        for j, sid in enumerate(station_ids):
            if sid in self.station_coordinates:
                lat, lon = self.station_coordinates[sid]
                
               
                if sid in self.station_ids:
                   
                    ax.scatter(lon, lat, c=EXISTING_COLOR, s=50, alpha=0.8,
                              edgecolor=EXISTING_EDGE, linewidth=0.5, zorder=4)
                else:
                    
                    ax.scatter(lon, lat, c=NEW_LINE_COLOR, s=50, alpha=0.8,
                              edgecolor=NEW_LINE_EDGE, linewidth=0.5, zorder=4)
        
        # Much bigger title
        title = f'{line_name}'
        if status:
            title += f' - {status}'
        ax.set_title(title, fontsize=24, fontweight='bold', pad=25)
        
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.grid(True, alpha=0.3)
        
       
        info_text = (
            f'Impacto Total: {total_impact:,.0f} viagens/dia  |  '
            f'Melhoria: {percent_improvement:.1f}%  |  '
            f'Por Estação: {impact_per_station:,.0f}  |  '
            f'Por km: {impact_per_km:,.0f}\n'
            f'Estações: {len(station_ids)} total ({len(new_stations)} novas, {len(connections)} conexões)  |  '
            f'Comprimento: {line_length_km:.1f} km'
        )
        
       
        ax.text(0.5, 0.02, info_text, transform=ax.transAxes,
               fontsize=14, verticalalignment='bottom', horizontalalignment='center',
               bbox=dict(boxstyle='round,pad=0.7', facecolor='white', alpha=0.95, 
                        edgecolor='white', linewidth=0), 
               fontweight='bold')
        
       
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=EXISTING_COLOR, 
                      markersize=10, markeredgecolor=EXISTING_EDGE, markeredgewidth=0.5, 
                      label='Estações Existentes'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=NEW_LINE_COLOR, 
                      markersize=10, markeredgecolor=NEW_LINE_EDGE, markeredgewidth=0.5, 
                      label='Novas Estações')
        ]
        ax.legend(handles=legend_elements, loc='upper right', framealpha=0.95, 
                 fontsize=12, edgecolor='white') 
        
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        
        
        if file_name is None:
           
            safe_name = line_name.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
            safe_name = safe_name.replace('ã', 'a').replace('á', 'a').replace('à', 'a')
            safe_name = safe_name.replace('í', 'i').replace('ó', 'o')
            file_name = f"{safe_name}.png"
        
        save_path = os.path.join(OUTPUT_DIR, file_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        viz_time = time.time() - viz_start
        
        total_time = time.time() - start_time
        print(f"   ✅ {file_name}: +{percent_improvement:.1f}% | Eval: {eval_time:.2f}s | Viz: {viz_time:.2f}s | Total: {total_time:.2f}s")
        
        return {
            'line_name': line_name,
            'file_name': file_name,
            'total_impact': total_impact,
            'percent_improvement': percent_improvement,
            'impact_per_station': impact_per_station,
            'impact_per_km': impact_per_km,
            'new_stations': len(new_stations),
            'connections': len(connections),
            'length_km': line_length_km,
            'status': status,
            'execution_time': total_time
        }

def generate_all_visualizations():
    """Generate individual visualizations for all lines."""
    
    print("\n" + "="*70)
    print("GENERATING INDIVIDUAL LINE VISUALIZATIONS")
    print("="*70)
    
    total_start_time = time.time()
    

    lines_with_status = {
        'Linha Laranja (L6)': {
            'stations': [18, 104, 50, 56, 44, 84, 133, 97, 73, 77, 40, 48, 1, 13, 89],
            'status': 'Linha em Construção',
            'file': 'linha_laranja_6.png'
        },
        'Linha Ouro (L17)': {
            'stations': [184, 30, 105, 25, 102, 22],
            'status': 'Linha em Construção',
            'file': 'linha_ouro_17.png'
        },
        'L17 - Ramal Congonhas': {
            'stations': [22, 33],
            'status': 'Ramal em Construção',
            'file': 'linha_ouro_17_congonhas.png'
        },
        'L17 - Ramal Washington Luís': {
            'stations': [102, 22, 53],
            'status': 'Ramal em Construção',
            'file': 'linha_ouro_17_washington.png'
        },
        'Expansão Verde (L2)': {
            'stations': [64, 81, 8, 107, 83, 46, 10, 72],
            'status': 'Expansão em Construção',
            'file': 'expansao_verde_2.png'
        },
       
        'Expansão Prata (L15) - Jd. Colonial': {
            'stations': [54, 15, 52, 207, 208, 209, 210],
            'status': 'Expansão em Construção',
            'file': 'expansao_prata_15_colonial.png'
        },
        'Expansão Amarela (L4)': {
            'stations': [112, 36, 100],
            'status': 'Expansão em Construção',
            'file': 'expansao_amarela_4.png'
        },
        'Expansão Lilás (L5)': {
            'stations': [27, 211, 212],
            'status': 'Expansão em Construção',
            'file': 'expansao_lilas_5.png'
        
        },
        'Linha Celeste (19)': {
            'stations': [ 213,214,215,216,217,218,219,220,221,222,223, 224,225, 88, 9],
           
            'status': 'Licitacão Ativa',
            'file': 'celeste_19.png'
        },
        'Linha Rosa (20)': {
            'stations': [230,132,231,232,233,43,235,236,237,39,239,240,95,243, 244,245,246,247,248,249],
            'status': 'Em elaboração de Projeto',
            'file': 'rosa_20.png'
        },
        'Linha Marrom (22)': {
            'stations': [251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,180,41,268,87],
            'status': 'Em elaboração de Projeto',
            'file': 'marrom_22.png'
        },
        'Linha Topazio (25)': {
            'stations': [ 295,296,297,298,299,300,301,302,303,304,305,188,307,308,309,26,311,312,313,314,315],
            'status': 'Em elaboração de Projeto',
            'file': 'marrom_22.png'
        },
        'Linha Violeta (16)': {
            'stations': [ 65,316,317, 68,318,319,320,321,322,323,324,325, 8, 326,327,328,329,330,331,332,333,334],
            'status': 'Priorizada pelo Governo',
            'file': 'violeta_16.png'
        }
    }
    
    # Initialize visualizer with timing
    print("\n⏳ Initializing system...")
    init_start = time.time()
    visualizer = IndividualLineVisualizer()
    init_time = time.time() - init_start
    print(f"   ✅ System initialized in {init_time:.2f} seconds")
    
    # Process each line
    results = []
    print(f"\n📊 Processing {len(lines_with_status)} lines...")
    print(f"   Output directory: {OUTPUT_DIR}\n")
    print("   Line Processing Times:")
    print("   " + "-"*60)
    
    processing_start = time.time()
    
    for line_name, line_info in lines_with_status.items():
        result = visualizer.evaluate_and_visualize_line(
            station_ids=line_info['stations'],
            line_name=line_name,
            status=line_info['status'],
            file_name=line_info['file']
        )
        results.append(result)
    
    processing_time = time.time() - processing_start
    
    # Create summary Excel with timing
    print("\n📊 Creating summary Excel...")
    excel_start = time.time()
    df = pd.DataFrame(results)
    
    # Reorder and rename columns for Excel
    df_excel = pd.DataFrame({
        'Nome da Linha': df['line_name'],
        'Status': df['status'],
        'Impacto Total (viagens/dia)': df['total_impact'].round(0),
        'Melhoria do Sistema (%)': df['percent_improvement'].round(2),
        'Impacto por Estação': df['impact_per_station'].round(0),
        'Impacto por km': df['impact_per_km'].round(0),
        'Novas Estações': df['new_stations'],
        'Conexões': df['connections'],
        'Comprimento (km)': df['length_km'].round(1),
        'Arquivo': df['file_name'],
        'Tempo (s)': df['execution_time'].round(2)
    })
    
    # Sort by impact
    df_excel = df_excel.sort_values('Melhoria do Sistema (%)', ascending=False)
    
    # Save Excel
    excel_path = os.path.join(RESULTS_DIR, 'analise_linhas_individuais.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_excel.to_excel(writer, sheet_name='Análise', index=False)
        
        # Format
        worksheet = writer.sheets['Análise']
        for col in worksheet.columns:
            max_length = 0
            column = col[0].column_letter
            for cell in col:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 40)
            worksheet.column_dimensions[column].width = adjusted_width
    
    excel_time = time.time() - excel_start
    print(f"   ✅ Excel saved in {excel_time:.2f}s: {excel_path}")
    
    # Total execution time
    total_time = time.time() - total_start_time
    
    # Summary with timing
    print("\n" + "="*70)
    print("EXECUTION TIME ANALYSIS")
    print("="*70)
    print(f"\n⏱️  Performance Metrics:")
    print(f"   System initialization: {init_time:.2f}s")
    print(f"   Lines processing: {processing_time:.2f}s")
    print(f"   Excel generation: {excel_time:.2f}s")
    print(f"   Total execution: {total_time:.2f}s")
    print(f"\n   Average per line: {processing_time/len(results):.2f}s")
    print(f"   Fastest line: {df['execution_time'].min():.2f}s")
    print(f"   Slowest line: {df['execution_time'].max():.2f}s")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n✅ Generated {len(results)} individual visualizations")
    print(f"   Location: {OUTPUT_DIR}")
    print(f"\n📊 Top 3 lines by impact:")
    for idx, row in df_excel.head(3).iterrows():
        print(f"   {row['Nome da Linha']}: +{row['Melhoria do Sistema (%)']}%")
    
    print("\n📁 Files created:")
    for result in results[:5]:  # Show first 5
        print(f"   {result['file_name']}")
    if len(results) > 5:
        print(f"   ... and {len(results)-5} more")
    
    return df_excel

if __name__ == "__main__":
    df = generate_all_visualizations()
    print("\n✅ All visualizations created successfully!")
    print(f"   Check folder: {OUTPUT_DIR}")