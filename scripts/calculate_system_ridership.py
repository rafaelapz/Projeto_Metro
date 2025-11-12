
from collections import defaultdict
from copy import deepcopy
import json
import numpy as np
import pandas as pd
from dados.linhas import *

def load_station_names():
    """Load station names from estacoes.csv"""
    estacoes_path = '/Users/ellazyngier/Documents/github/tccII/scripts/dados/estacoes.csv'
    
    try:
        print(f"Loading station names from: {estacoes_path}")
        df_estacoes = pd.read_csv(estacoes_path)
        
       
        station_names = {}
        for _, row in df_estacoes.iterrows():
            codigo = str(row['codigo_estacao'])
            nome = row['nome']
            station_names[codigo] = nome
        
        print(f"Loaded {len(station_names)} station names")
        return station_names
    except Exception as e:
        print(f"Warning: Could not load station names: {e}")
        return {}

def load_data():
    """Load the JSON data and numpy matrix"""
    # Load station percentage zones
    json_path = '/Users/ellazyngier/Documents/github/tccII/scripts/dados/station_zones_882.json'
    
    # Use the CLEAN matrix file (no headers, no totals, 527x527)
    npy_path = '/Users/ellazyngier/Documents/github/tccII/scripts/dados/travel_matrix_VIAGENS_MOTORIZADAS_SOMENTE.npy'
    
    print(f"Loading station zones from: {json_path}")
    with open(json_path, 'r') as f:
        station_data = json.load(f)
    
    # Load the CLEAN travel matrix (already processed, no headers)
    print(f"Loading CLEAN travel matrix from: {npy_path}")
    travel_matrix = np.load(npy_path)
    
    print(f"Matrix shape: {travel_matrix.shape}")
    
    return station_data, travel_matrix

def calculate_station_pair_flow(station_a_zones, station_b_zones, travel_matrix):

    total_flow = 0
    
    for zone_a, pct_a in station_a_zones.items():
        for zone_b, pct_b in station_b_zones.items():
            # Convert zone IDs (strings "1"-"527") to matrix indices (0-526)
            zone_a_idx = int(zone_a) - 1  # Zone "1" → index 0
            zone_b_idx = int(zone_b) - 1  # Zone "1" → index 0
            
            # Check bounds (should always be valid for zones 1-527)
            if 0 <= zone_a_idx < 527 and 0 <= zone_b_idx < 527:
                # Flow from A to B
                trips_a_to_b = travel_matrix[zone_a_idx, zone_b_idx]
                flow_a_to_b = trips_a_to_b * (pct_a / 100) * (pct_b / 100)
                
                # Flow from B to A
                trips_b_to_a = travel_matrix[zone_b_idx, zone_a_idx]
                flow_b_to_a = trips_b_to_a * (pct_b / 100) * (pct_a / 100)
                
                total_flow += flow_a_to_b + flow_b_to_a
    
    return total_flow

def calculate_network_ridership(station_zones_data, travel_matrix):
    # Find which zones have metro access
    zones_with_access = set()
    zone_total_coverage = {}
    
    for station_id, zones in station_zones_data.items():
        for zone_id, percentage in zones.items():
            zones_with_access.add(zone_id)
            if zone_id not in zone_total_coverage:
                zone_total_coverage[zone_id] = 0
            zone_total_coverage[zone_id] += percentage
    
    # Calculate metrics
    total_possible_trips = np.sum(travel_matrix)
    metro_accessible_trips = 0
    served_to_served_trips = 0
    
    # Iterate through matrix (0-indexed)
    for i in range(527):  # Matrix rows 0-526
        for j in range(527):  # Matrix columns 0-526
            trips = travel_matrix[i, j]
            if trips > 0:
                # Convert matrix indices to zone IDs (add 1)
                origin_zone = str(i + 1)  # Matrix index 0 → Zone "1"
                dest_zone = str(j + 1)    # Matrix index 0 → Zone "1"
                
                origin_has_metro = origin_zone in zones_with_access
                dest_has_metro = dest_zone in zones_with_access
                
                if origin_has_metro and dest_has_metro:
                    served_to_served_trips += trips
                    origin_coverage = min(zone_total_coverage.get(origin_zone, 0) / 100, 1.0)
                    dest_coverage = min(zone_total_coverage.get(dest_zone, 0) / 100, 1.0)
                    metro_accessible_trips += trips * origin_coverage * dest_coverage
    
    return {
        'total_possible_trips': total_possible_trips,
        'served_to_served_trips': served_to_served_trips,
        'metro_accessible_trips': metro_accessible_trips,
        'network_coverage': (served_to_served_trips / total_possible_trips * 100) if total_possible_trips > 0 else 0,
        'effective_coverage': (metro_accessible_trips / total_possible_trips * 100) if total_possible_trips > 0 else 0,
        'zones_with_access': len(zones_with_access),
        'avg_zone_coverage': np.mean(list(zone_total_coverage.values())) if zone_total_coverage else 0
    }

def calculate_new_station_impact(new_station_zones, existing_station_data, travel_matrix):
   
    # Calculate baseline metrics
    baseline_metrics = calculate_network_ridership(existing_station_data, travel_matrix)
    
    # Create updated station data with new station
    updated_station_data = deepcopy(existing_station_data)
    updated_station_data['NEW_STATION'] = new_station_zones
    
    # Calculate metrics with new station
    new_metrics = calculate_network_ridership(updated_station_data, travel_matrix)
    
    # Calculate improvements
    impact = {
        'baseline_ridership': baseline_metrics['metro_accessible_trips'],
        'new_ridership': new_metrics['metro_accessible_trips'],
        'ridership_increase': new_metrics['metro_accessible_trips'] - baseline_metrics['metro_accessible_trips'],
        'ridership_increase_pct': ((new_metrics['metro_accessible_trips'] - baseline_metrics['metro_accessible_trips']) / 
                                   baseline_metrics['metro_accessible_trips'] * 100) if baseline_metrics['metro_accessible_trips'] > 0 else 0,
        'new_zones_served': new_metrics['zones_with_access'] - baseline_metrics['zones_with_access'],
        'coverage_improvement': new_metrics['network_coverage'] - baseline_metrics['network_coverage']
    }
    
    return impact

def calculate_total_network_trips(station_zones, travel_matrix, all_stations):

    total_metro_trips = 0
    stations_with_data = [s for s in all_stations if s in station_zones]
    
    # Calculate trips between all pairs of stations
    for i, station_a in enumerate(stations_with_data):
        for station_b in stations_with_data[i:]:  # Start from i to avoid double counting
            if station_a == station_b:
                # Trips within same station catchment area
                flow = calculate_station_pair_flow(
                    station_zones[station_a],
                    station_zones[station_a],
                    travel_matrix
                ) / 2  # Divide by 2 since we're counting both directions in the same zone
            else:
                # Trips between different stations
                flow = calculate_station_pair_flow(
                    station_zones[station_a],
                    station_zones[station_b],
                    travel_matrix
                )
            
            total_metro_trips += flow
    
    return total_metro_trips

# CORRECT way to calculate station activity
def calculate_station_activity(station_zones, travel_matrix, all_station_zones):
    total = 0
    # Sum flows to/from all other stations
    for other_station in all_station_zones.values():
        if other_station != station_zones:
            flow = calculate_station_pair_flow(station_zones, other_station, travel_matrix)
            total += flow
    # Add within-station trips
    within = calculate_station_pair_flow(station_zones, station_zones, travel_matrix)
    total += within / 2  # Avoid double-counting
    return total


def rank_stations_by_boardings(station_zones, travel_matrix, all_stations, station_names, top_n=20):

    print("\n" + "=" * 80)
    print(f"TOP {top_n} ESTAÇÕES POR NÚMERO DE EMBARQUES")
    print("=" * 80)
    
    # Calculate boardings for each station
    station_boardings_dict = {}
    missing_data = []
    
    for station_id in all_stations:
        if station_id in station_zones:
            boardings = calculate_station_boardings(station_zones[station_id], travel_matrix)
            station_boardings_dict[station_id] = boardings
        else:
            missing_data.append(station_id)
    
    # Sort stations by boardings
    sorted_stations = sorted(station_boardings_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Display top N
    print(f"\n{'Rank':<6} {'Código':<8} {'Nome da Estação':<35} {'Embarques':<15}")
    print("-" * 80)
    
    for rank, (station_id, boardings) in enumerate(sorted_stations[:top_n], 1):
        station_name = station_names.get(station_id, "Nome desconhecido")
        print(f"{rank:<6} {station_id:<8} {station_name:<35} {boardings:>14,.0f}")
    
    # Statistics
    if station_boardings_dict:
        total_boardings = sum(station_boardings_dict.values())
        avg_boardings = total_boardings / len(station_boardings_dict)
        
        print(f"\nEstatísticas de embarques:")
        print(f"  Total de embarques: {total_boardings:,.0f}")
        print(f"  Média por estação: {avg_boardings:,.0f}")
        print(f"  Estações analisadas: {len(station_boardings_dict)}")
        
        if missing_data:
            print(f"  Estações sem dados: {len(missing_data)}")
    
    return sorted_stations

def main():
    print("Análise Simplificada de Viagens no Sistema")
    print("=" * 60)
    
    # Load data
    print("Carregando dados...")
    station_zones, travel_matrix = load_data()
    station_names = load_station_names()
    
    # Get all unique stations from the lines
    all_stations = set()
    for line in current_and_express_lines:
        all_stations.update(str(s) for s in line)
    
    print(f"Total de estações na rede: {len(all_stations)}")
    
    # Count stations with zone data
    stations_with_data = [s for s in all_stations if s in station_zones]
    print(f"Estações com dados de zona: {len(stations_with_data)}")
    
    # Missing stations
    missing_zones = [s for s in all_stations if s not in station_zones]
    if missing_zones:
        print(f"\nAviso: {len(missing_zones)} estações sem dados de zona:")
        for station in sorted(missing_zones, key=int)[:10]:  # Show first 10
            station_name = station_names.get(station, "Nome desconhecido")
            print(f"  - Estação {station} ({station_name})")
        if len(missing_zones) > 10:
            print(f"  ... e mais {len(missing_zones) - 10} estações")
    
    print("\n" + "=" * 80)
    print("RESULTADOS PRINCIPAIS")
    print("=" * 80)
    
    # 1. Calculate total trips in the entire matrix
    total_all_trips = np.sum(travel_matrix)
    print(f"\n1. TOTAL DE VIAGENS NA MATRIZ (todo o sistema):")
    print(f"   {total_all_trips:,.0f} viagens")
    
    # Additional verification methods
    print(f"\n   Verificação alternativa:")
    print(f"   - Soma usando .sum(): {travel_matrix.sum():,.0f}")
    print(f"   - Soma usando flatten: {np.sum(travel_matrix.flatten()):,.0f}")
    print(f"   - Soma linha por linha: {sum(np.sum(travel_matrix[i,:]) for i in range(527)):,.0f}")
    
    # Check if matrix is symmetric (should be for O-D matrix)
    if np.allclose(travel_matrix, travel_matrix.T):
        print(f"   - Matriz é simétrica ✓")
    else:
        print(f"   - AVISO: Matriz NÃO é simétrica!")
        diff_count = np.sum(travel_matrix != travel_matrix.T)
        print(f"     {diff_count} células diferem da transposta")
    
    # 2. Calculate trips between metro stations
    print(f"\n2. VIAGENS ENTRE ESTAÇÕES DO METRÔ:")
    print("   Calculando fluxos entre pares de estações...")
    
    # Quick sanity check on station zones data
    all_zone_ids = set()
    for station_zones_dict in station_zones.values():
        all_zone_ids.update(station_zones_dict.keys())
    
    zone_ids_as_ints = sorted([int(z) for z in all_zone_ids if z.isdigit()])
    if zone_ids_as_ints:
        print(f"   Zonas cobertas pelas estações: {len(zone_ids_as_ints)} zonas")
        print(f"   Faixa de zonas: {min(zone_ids_as_ints)} até {max(zone_ids_as_ints)}")
    
    total_metro_trips = calculate_total_network_trips(station_zones, travel_matrix, all_stations)
    
    print(f"   {total_metro_trips:,.0f} viagens")
    
    # Calculate percentage
    percentage_served = (total_metro_trips / total_all_trips * 100) if total_all_trips > 0 else 0
    
    print(f"\n3. COBERTURA DO SISTEMA:")
    print(f"   O metrô atende {percentage_served:.2f}% de todas as viagens na matriz")
    print(f"   Viagens não atendidas: {total_all_trips - total_metro_trips:,.0f} ({100 - percentage_served:.2f}%)")
    
    # 4. Rank stations by boardings
    boarding_ranking = rank_stations_by_boardings(station_zones, travel_matrix, all_stations, station_names, top_n=20)
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()