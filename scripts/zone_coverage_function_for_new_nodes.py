

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import json
from typing import List, Tuple, Dict
import time

class NoOverlapLineFinder:
    def __init__(self, 
                 travel_matrix_path: str, 
                 existing_stations_csv: str,
                 zones_shapefile_path: str,
                 station_zones_json: str,
                 catchment_radius: int = 882):
        
        print("Loading data...")
        self.travel_matrix = np.load(travel_matrix_path)
        self.zones_gdf = gpd.read_file(zones_shapefile_path)
        self.catchment_radius = catchment_radius
        self.min_station_distance = catchment_radius * 2 / 1000  # 1.764 km
        self.max_station_distance = 3.0  # km
        
        # Load existing stations
        df = pd.read_csv(existing_stations_csv)
        self.existing_stations = [(row['latitude'], row['longitude']) 
                                 for _, row in df.iterrows()]
        
        # Load existing station zones for faster calculation
        with open(station_zones_json, 'r') as f:
            self.existing_station_zones = json.load(f)
        
        print(f"Loaded {len(self.existing_stations)} existing stations")
        print(f"Minimum station distance: {self.min_station_distance:.3f} km (no overlaps)")
        
        # Pre-calculate zone importance for scoring
        self.zone_importance = np.sum(self.travel_matrix, axis=0) + \
                              np.sum(self.travel_matrix, axis=1)
    
    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance in km between two points"""
        R = 6371  # Earth radius
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * \
            np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
        return R * 2 * np.arcsin(np.sqrt(a))
    
    def get_station_zones(self, lat: float, lon: float) -> Dict[str, float]:
        """
        Calculate which zones a station covers (882m radius)
        Returns {zone_id: coverage_percentage}
        """
        # Create station point and buffer
        station_point = Point(lon, lat)
        station_gdf = gpd.GeoDataFrame([{'id': 'NEW'}], 
                                       geometry=[station_point],
                                       crs='EPSG:4326')
        
        # Project to UTM for accurate distance
        station_projected = station_gdf.to_crs('EPSG:32723')
        zones_projected = self.zones_gdf.to_crs('EPSG:32723')
        
        # Create buffer
        station_buffer = station_projected.geometry[0].buffer(self.catchment_radius)
        
        # Find zone intersections
        zone_coverage = {}
        for idx, zone in zones_projected.iterrows():
            intersection = station_buffer.intersection(zone.geometry)
            if not intersection.is_empty:
                coverage_pct = (intersection.area / zone.geometry.area) * 100
                if coverage_pct > 0.1:
                    zone_id = str(idx + 1)
                    zone_coverage[zone_id] = round(coverage_pct, 2)
        
        return zone_coverage
    
    def calculate_network_ridership(self, new_stations: List[Tuple[float, float]]) -> float:

        # Combine existing and new station zones
        all_station_zones = dict(self.existing_station_zones)
        
        # Add new stations
        for i, (lat, lon) in enumerate(new_stations):
            zones = self.get_station_zones(lat, lon)
            all_station_zones[f'NEW_{i}'] = zones
        
        # Get maximum coverage for each zone (should be no overlaps anyway)
        zone_coverage = {}
        for station_zones in all_station_zones.values():
            for zone_id, coverage in station_zones.items():
                if zone_id not in zone_coverage:
                    zone_coverage[zone_id] = coverage
                else:
                    # Shouldn't happen with no overlaps, but just in case
                    zone_coverage[zone_id] = max(zone_coverage[zone_id], coverage)
        
        # Calculate total ridership
        total_ridership = 0
        zones_with_access = list(zone_coverage.keys())
        
        for zone_a in zones_with_access:
            for zone_b in zones_with_access:
                zone_a_idx = int(zone_a) - 1
                zone_b_idx = int(zone_b) - 1
                
                if 0 <= zone_a_idx < 527 and 0 <= zone_b_idx < 527:
                    trips = self.travel_matrix[zone_a_idx, zone_b_idx]
                    coverage_a = zone_coverage[zone_a] / 100
                    coverage_b = zone_coverage[zone_b] / 100
                    total_ridership += trips * coverage_a * coverage_b
        
        return total_ridership
    
    def is_valid_location(self, lat: float, lon: float, 
                         existing_line: List[Tuple[float, float]]) -> bool:

        # Check distance from existing network stations
        for ex_lat, ex_lon in self.existing_stations:
            if self.calculate_distance(lat, lon, ex_lat, ex_lon) < self.min_station_distance:
                return False
        
        # Check distance from stations in current line
        for line_lat, line_lon in existing_line:
            if self.calculate_distance(lat, lon, line_lat, line_lon) < self.min_station_distance:
                return False
        
        return True
    
    def find_optimal_line(self, num_stations: int = 5) -> List[Tuple[float, float]]:

        print(f"\nFinding optimal {num_stations}-station line (no overlaps)...")
        
        # São Paulo metro area bounds
        lat_min, lat_max = -23.70, -23.45
        lon_min, lon_max = -46.75, -46.55
        
        # Generate candidate grid
        candidates = []
        for lat in np.linspace(lat_min, lat_max, 40):
            for lon in np.linspace(lon_min, lon_max, 40):
                lat_val = float(lat)
                lon_val = float(lon)
                # Only include if far enough from existing stations
                if self.is_valid_location(lat_val, lon_val, []):
                    candidates.append((lat_val, lon_val))
        
        print(f"Generated {len(candidates)} valid candidates (no overlap with existing)")
        
        if len(candidates) == 0:
            print("ERROR: No valid locations found! Check bounds and existing station density.")
            return []
        
        # Calculate baseline ridership (existing network only)
        baseline_ridership = self.calculate_network_ridership([])
        print(f"Baseline ridership (existing network): {baseline_ridership:,.0f}")
        
        # Find best starting point
        print("\nEvaluating starting points...")
        best_start = None
        best_gain = 0
        
        for i, candidate in enumerate(candidates[:100]):  # Test first 100
            if i % 20 == 0:
                print(f"  Testing candidate {i+1}...")
            
            ridership = self.calculate_network_ridership([candidate])
            gain = ridership - baseline_ridership
            
            if gain > best_gain:
                best_gain = gain
                best_start = candidate
                print(f"    New best: {candidate} with gain +{gain:,.0f}")
        
        if best_start is None:
            print("ERROR: No valid starting point found!")
            return []
        
        print(f"\nBest starting point: ({best_start[0]:.4f}, {best_start[1]:.4f})")
        print(f"Ridership gain: +{best_gain:,.0f}")
        
        # Build line greedily
        line = [best_start]
        current_ridership = baseline_ridership + best_gain
        
        for station_num in range(2, num_stations + 1):
            print(f"\nAdding station {station_num}...")
            
            # Find valid candidates (right distance from last station, no overlaps)
            last_lat, last_lon = line[-1]
            valid_candidates = []
            
            for lat, lon in candidates:
                if (lat, lon) in line:
                    continue
                
                # Check distance from last station
                dist_from_last = self.calculate_distance(lat, lon, last_lat, last_lon)
                if not (self.min_station_distance <= dist_from_last <= self.max_station_distance):
                    continue
                
                # Check no overlap with any station
                if self.is_valid_location(lat, lon, line):
                    valid_candidates.append((lat, lon))
            
            print(f"  Found {len(valid_candidates)} valid candidates")
            
            if not valid_candidates:
                print("  No valid candidates found! Stopping here.")
                break
            
            # Test each valid candidate
            best_next = None
            best_next_ridership = current_ridership
            
            for candidate in valid_candidates[:50]:  # Test up to 50
                test_line = line + [candidate]
                ridership = self.calculate_network_ridership(test_line)
                
                if ridership > best_next_ridership:
                    best_next_ridership = ridership
                    best_next = candidate
            
            if best_next:
                line.append(best_next)
                gain = best_next_ridership - current_ridership
                current_ridership = best_next_ridership
                
                dist = self.calculate_distance(last_lat, last_lon, best_next[0], best_next[1])
                print(f"  Added: ({best_next[0]:.4f}, {best_next[1]:.4f})")
                print(f"  Distance from previous: {dist:.2f} km")
                print(f"  Ridership gain: +{gain:,.0f}")
                print(f"  Total ridership: {current_ridership:,.0f}")
        
        return line


# Main execution
if __name__ == "__main__":
    finder = NoOverlapLineFinder(
        travel_matrix_path='/Users/ellazyngier/Documents/github/tccII/scripts/dados/travel_matrix_VIAGENS_MOTORIZADAS_SOMENTE.npy',
        existing_stations_csv='/Users/ellazyngier/Documents/github/tccII/scripts/dados/estacoes.csv',
        zones_shapefile_path='/Users/ellazyngier/Documents/github/tccII/Site_190225/002_Site Metro Mapas_190225/Shape/Zonas_2023.shp',
        station_zones_json='/Users/ellazyngier/Documents/github/tccII/scripts/dados/station_zones_882.json'
    )
    
    start_time = time.time()
    optimal_line = finder.find_optimal_line(5)
    elapsed = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"Optimization completed in {elapsed:.1f} seconds")
    print(f"{'='*60}")
    
    if optimal_line:
        print("\nOPTIMAL LINE FOUND:")
        total_length = 0
        for i, (lat, lon) in enumerate(optimal_line):
            if i > 0:
                prev_lat, prev_lon = optimal_line[i-1]
                dist = finder.calculate_distance(lat, lon, prev_lat, prev_lon)
                total_length += dist
                print(f"  Station {i+1}: ({lat:.4f}, {lon:.4f}) - {dist:.2f} km from previous")
            else:
                print(f"  Station 1: ({lat:.4f}, {lon:.4f})")
        
        if len(optimal_line) > 1:
            print(f"\nLine statistics:")
            print(f"  Total length: {total_length:.2f} km")
            print(f"  Average spacing: {total_length/(len(optimal_line)-1):.2f} km")
            
            # Calculate total gain
            baseline = finder.calculate_network_ridership([])
            final = finder.calculate_network_ridership(optimal_line)
            print(f"  Total ridership gain: +{final-baseline:,.0f} trips")