

import numpy as np
import pandas as pd
import sys
import os
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns

# Add the path to import linhas.py
sys.path.append('/Users/ellazyngier/Documents/github/tccII/scripts/dados/')
from dados.linhas import (
    azul1, verde2, vermelha3, amarela4, lilas5,  diamante8,
    esmeralda9, turquesa10, coral11, safira12, jade13, prata15,
)

class MetroLineSpacingAnalyzer:
    def __init__(self, existing_stations_csv: str):
        """Initialize with station data"""
        print("Loading station data...")
        
        # Load existing stations
        self.stations_df = pd.read_csv(existing_stations_csv)
        
        # Create a dictionary mapping station code to coordinates
        self.station_coords = {}
        for _, row in self.stations_df.iterrows():
            # Note: station codes in linhas.py are 1-indexed, but DataFrame index might be 0-indexed
            # Using codigo_estacao as the key
            self.station_coords[row['codigo_estacao']] = {
                'lat': row['latitude'],
                'lon': row['longitude'],
                'name': row.get('nome', f"Station {row['codigo_estacao']}")
            }
        
        print(f"Loaded {len(self.station_coords)} stations")
        
        # Define lines with their names
        self.lines = {
            'Linha 1 - Azul': azul1,
            'Linha 2 - Verde': verde2,
            'Linha 3 - Vermelha': vermelha3,
            'Linha 4 - Amarela': amarela4,
            'Linha 5 - Lilás': lilas5,
            'Linha 8 - Diamante': diamante8,
            'Linha 9 - Esmeralda': esmeralda9,
            'Linha 10 - Turquesa': turquesa10,
            'Linha 11 - Coral': coral11,
            'Linha 12 - Safira': safira12,
            'Linha 13 - Jade': jade13,
            'Linha 15 - Prata': prata15,
        }
    
    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance in km using Haversine formula"""
        R = 6371  # Earth's radius in km
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * \
            np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
        return R * 2 * np.arcsin(np.sqrt(a))
    
    def analyze_line_spacing(self, line_stations: List[int]) -> Dict:
        """Analyze spacing for a single line"""
        distances = []
        station_pairs = []
        missing_stations = []
        
        # Calculate distances between consecutive stations
        for i in range(len(line_stations) - 1):
            station1_id = line_stations[i]
            station2_id = line_stations[i + 1]
            
            # Check if both stations exist in our data
            if station1_id not in self.station_coords:
                missing_stations.append(station1_id)
                continue
            if station2_id not in self.station_coords:
                missing_stations.append(station2_id)
                continue
            
            # Get coordinates
            lat1 = self.station_coords[station1_id]['lat']
            lon1 = self.station_coords[station1_id]['lon']
            name1 = self.station_coords[station1_id]['name']
            
            lat2 = self.station_coords[station2_id]['lat']
            lon2 = self.station_coords[station2_id]['lon']
            name2 = self.station_coords[station2_id]['name']
            
            # Calculate distance
            distance = self.calculate_distance(lat1, lon1, lat2, lon2)
            distances.append(distance)
            station_pairs.append((name1, name2, distance))
        
        if not distances:
            return {
                'min_distance': None,
                'max_distance': None,
                'avg_distance': None,
                'std_distance': None,
                'num_segments': 0,
                'missing_stations': list(set(missing_stations)),
                'station_pairs': []
            }
        
        return {
            'min_distance': min(distances),
            'max_distance': max(distances),
            'avg_distance': np.mean(distances),
            'std_distance': np.std(distances),
            'num_segments': len(distances),
            'min_pair': station_pairs[distances.index(min(distances))],
            'max_pair': station_pairs[distances.index(max(distances))],
            'all_distances': distances,
            'station_pairs': station_pairs,
            'missing_stations': list(set(missing_stations))
        }
    
    def analyze_all_lines(self) -> pd.DataFrame:
        """Analyze all lines and return summary DataFrame"""
        results = []
        
        print("\n" + "="*80)
        print("ANALYZING STATION SPACING FOR ALL METRO LINES")
        print("="*80)
        
        for line_name, line_stations in self.lines.items():
            analysis = self.analyze_line_spacing(line_stations)
            
            if analysis['num_segments'] > 0:
                print(f"\n{line_name}:")
                print(f"  Stations: {len(line_stations)}")
                print(f"  Segments analyzed: {analysis['num_segments']}")
                print(f"  Min distance: {analysis['min_distance']:.3f} km")
                print(f"    Between: {analysis['min_pair'][0]} - {analysis['min_pair'][1]}")
                print(f"  Max distance: {analysis['max_distance']:.3f} km")
                print(f"    Between: {analysis['max_pair'][0]} - {analysis['max_pair'][1]}")
                print(f"  Average distance: {analysis['avg_distance']:.3f} km")
                print(f"  Std deviation: {analysis['std_distance']:.3f} km")
                
                if analysis['missing_stations']:
                    print(f"  ⚠️  Missing station data for IDs: {analysis['missing_stations']}")
                
                results.append({
                    'Line': line_name,
                    'Total Stations': len(line_stations),
                    'Segments Analyzed': analysis['num_segments'],
                    'Min Distance (km)': round(analysis['min_distance'], 3),
                    'Max Distance (km)': round(analysis['max_distance'], 3),
                    'Avg Distance (km)': round(analysis['avg_distance'], 3),
                    'Std Dev (km)': round(analysis['std_distance'], 3),
                    'Min Pair': f"{analysis['min_pair'][0][:15]} - {analysis['min_pair'][1][:15]}",
                    'Max Pair': f"{analysis['max_pair'][0][:15]} - {analysis['max_pair'][1][:15]}"
                })
            else:
                print(f"\n{line_name}: NO DATA AVAILABLE")
                if analysis['missing_stations']:
                    print(f"  Missing all station data for IDs: {analysis['missing_stations']}")
        
        return pd.DataFrame(results)
    
    def plot_spacing_distribution(self, output_path: str = 'metro_spacing_distribution.png'):
        """Create visualization of spacing distributions"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Collect all distances by line
        all_line_distances = {}
        for line_name, line_stations in self.lines.items():
            analysis = self.analyze_line_spacing(line_stations)
            if analysis['num_segments'] > 0:
                all_line_distances[line_name] = analysis['all_distances']
        
        # 1. Box plot of distances by line
        ax1 = axes[0, 0]
        data_for_box = []
        labels_for_box = []
        for line_name, distances in all_line_distances.items():
            data_for_box.append(distances)
            # Shorten line names for readability
            short_name = line_name.split(' - ')[0] if ' - ' in line_name else line_name
            labels_for_box.append(short_name)
        
        bp = ax1.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax1.set_ylabel('Distance (km)')
        ax1.set_title('Station Spacing Distribution by Line')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=1.8, color='r', linestyle='--', alpha=0.5, label='Ideal (1.8 km)')
        ax1.legend()
        
        # 2. Histogram of all distances
        ax2 = axes[0, 1]
        all_distances = []
        for distances in all_line_distances.values():
            all_distances.extend(distances)
        
        ax2.hist(all_distances, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax2.axvline(x=np.mean(all_distances), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(all_distances):.2f} km')
        ax2.axvline(x=np.median(all_distances), color='green', linestyle='--',
                   label=f'Median: {np.median(all_distances):.2f} km')
        ax2.set_xlabel('Distance (km)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Overall Distribution of Station Spacing')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Min vs Max distances by line
        ax3 = axes[1, 0]
        min_distances = []
        max_distances = []
        line_names_short = []
        
        for line_name, line_stations in self.lines.items():
            analysis = self.analyze_line_spacing(line_stations)
            if analysis['num_segments'] > 0:
                min_distances.append(analysis['min_distance'])
                max_distances.append(analysis['max_distance'])
                short_name = line_name.split(' - ')[0] if ' - ' in line_name else line_name
                line_names_short.append(short_name)
        
        x = np.arange(len(line_names_short))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, min_distances, width, label='Min Distance', color='green', alpha=0.7)
        bars2 = ax3.bar(x + width/2, max_distances, width, label='Max Distance', color='red', alpha=0.7)
        
        ax3.set_ylabel('Distance (km)')
        ax3.set_title('Minimum and Maximum Station Spacing by Line')
        ax3.set_xticks(x)
        ax3.set_xticklabels(line_names_short, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Average spacing with error bars (std dev)
        ax4 = axes[1, 1]
        avg_distances = []
        std_distances = []
        
        for line_name, line_stations in self.lines.items():
            analysis = self.analyze_line_spacing(line_stations)
            if analysis['num_segments'] > 0:
                avg_distances.append(analysis['avg_distance'])
                std_distances.append(analysis['std_distance'])
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(line_names_short)))
        bars = ax4.bar(x, avg_distances, yerr=std_distances, capsize=5, 
                      color=colors, alpha=0.7, edgecolor='black')
        
        ax4.set_ylabel('Distance (km)')
        ax4.set_title('Average Station Spacing with Standard Deviation')
        ax4.set_xticks(x)
        ax4.set_xticklabels(line_names_short, rotation=45, ha='right')
        ax4.axhline(y=1.8, color='r', linestyle='--', alpha=0.5, label='Ideal spacing (1.8 km)')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('São Paulo Metro Station Spacing Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Visualization saved to {output_path}")
        plt.show()
    
    def get_extreme_cases(self, n: int = 5) -> Dict:
        """Get the n smallest and largest distances across all lines"""
        all_segments = []
        
        for line_name, line_stations in self.lines.items():
            analysis = self.analyze_line_spacing(line_stations)
            for pair in analysis['station_pairs']:
                all_segments.append({
                    'line': line_name,
                    'from': pair[0],
                    'to': pair[1],
                    'distance': pair[2]
                })
        
        # Sort by distance
        all_segments.sort(key=lambda x: x['distance'])
        
        return {
            'shortest': all_segments[:n],
            'longest': all_segments[-n:]
        }


# Main execution
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = MetroLineSpacingAnalyzer(
        existing_stations_csv='/Users/ellazyngier/Documents/github/tccII/scripts/dados/estacoes.csv'
    )
    
    # Analyze all lines
    summary_df = analyzer.analyze_all_lines()
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(summary_df.to_string(index=False))
    
    # Save to CSV
    summary_df.to_csv('metro_line_spacing_summary.csv', index=False)
    print("\n✓ Summary saved to metro_line_spacing_summary.csv")
    
    # Get extreme cases
    print("\n" + "="*80)
    print("EXTREME CASES")
    print("="*80)
    
    extremes = analyzer.get_extreme_cases(n=5)
    
    print("\n5 SHORTEST DISTANCES:")
    for i, segment in enumerate(extremes['shortest'], 1):
        print(f"  {i}. {segment['distance']:.3f} km: {segment['from']} → {segment['to']}")
        print(f"     ({segment['line']})")
    
    print("\n5 LONGEST DISTANCES:")
    for i, segment in enumerate(extremes['longest'], 1):
        print(f"  {i}. {segment['distance']:.3f} km: {segment['from']} → {segment['to']}")
        print(f"     ({segment['line']})")
    
    # Create visualizations
    analyzer.plot_spacing_distribution()
    
    # Calculate overall statistics
    all_distances = []
    for line_name, line_stations in analyzer.lines.items():
        analysis = analyzer.analyze_line_spacing(line_stations)
        if analysis['num_segments'] > 0:
            all_distances.extend(analysis['all_distances'])
    
    if all_distances:
        print("\n" + "="*80)
        print("OVERALL NETWORK STATISTICS")
        print("="*80)
        print(f"Total segments analyzed: {len(all_distances)}")
        print(f"Overall minimum distance: {min(all_distances):.3f} km")
        print(f"Overall maximum distance: {max(all_distances):.3f} km")
        print(f"Overall average distance: {np.mean(all_distances):.3f} km")
        print(f"Overall median distance: {np.median(all_distances):.3f} km")
        print(f"Overall std deviation: {np.std(all_distances):.3f} km")
        print(f"25th percentile: {np.percentile(all_distances, 25):.3f} km")
        print(f"75th percentile: {np.percentile(all_distances, 75):.3f} km")