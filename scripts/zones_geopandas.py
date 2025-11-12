#TOO MANY THINGS ARE GETTING PRINTED ON THIS CODE - EXPORT TO TXT TO SEE IT ALL 
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import numpy as np

# 1. Load your zones shapefile
zonas = gpd.read_file("/Users/ellazyngier/Documents/github/tccII/Site_190225/002_Site Metro Mapas_190225/Shape/Zonas_2023.shp")

# 2. Load your CSV points with lat/lon
points_df = pd.read_csv("/Users/ellazyngier/Documents/github/tccII/scripts/dados/estacoes.csv")

# 3. Convert lat/lon to Shapely Points and then GeoDataFrame
geometry = [Point(xy) for xy in zip(points_df['longitude'], points_df['latitude'])]
points_gdf = gpd.GeoDataFrame(points_df, geometry=geometry)

# 4. Set CRS to WGS84 for lat/lon data
points_gdf.set_crs(epsg=4326, inplace=True)

# 5. Transform to a projected CRS for accurate distance calculations
target_crs = 'EPSG:32723'  # UTM Zone 23S
zonas_projected = zonas.to_crs(target_crs)
points_projected = points_gdf.to_crs(target_crs)

# 6. Calculate total area of each zone (in square meters)
zonas_projected['total_area_m2'] = zonas_projected.geometry.area

# 7. Create 400m buffer circles around each point
buffer_distance = 882  # meters
points_projected['circle'] = points_projected.geometry.buffer(buffer_distance)

# 8. NEW: Function to handle overlapping circles and divide overlapped areas
def create_non_overlapping_territories(points_gdf):

    print("Processing overlapping areas...")
    
    territories = []
    total_stations = len(points_gdf)
    
    # First, create a mapping of all circles using codigo_estacao
    circles = {}
    for idx, station_row in points_gdf.iterrows():
        codigo = station_row['codigo_estacao']  # Use codigo_estacao instead of idx
        circles[codigo] = {
            'circle': station_row['circle'],
            'df_index': idx  # Keep track of original dataframe index
        }
    
    for idx, station_row in points_gdf.iterrows():
        codigo = station_row['codigo_estacao']  # Use codigo_estacao
        print(f"Processing station {codigo}...")
        station_circle = station_row['circle']
        station_name = f"STATION_{codigo}"  # Use codigo_estacao in name
        
        # Find all overlapping stations using codigo_estacao
        overlapping_stations = []
        for other_codigo in circles:
            if codigo != other_codigo:
                other_circle = circles[other_codigo]['circle']
                try:
                    if station_circle.intersects(other_circle):
                        overlapping_stations.append(other_codigo)
                except Exception as e:
                    print(f"Warning: Error checking intersection between {codigo} and {other_codigo}: {e}")
                    continue
        
        # If no overlaps, keep the full circle
        if not overlapping_stations:
            territories.append({
                'station_codigo': codigo,  # Use codigo_estacao
                'station_name': station_name,
                'df_index': idx,  # Keep original index for reference
                'territory': station_circle,
                'original_area': station_circle.area,
                'final_area': station_circle.area,
                'exclusive_area': station_circle.area,
                'shared_areas': []
            })
            continue
        
        # Start with the full circle as exclusive area
        try:
            exclusive_area = station_circle
            shared_areas = []
            all_overlaps_to_subtract = []
            
            # Calculate overlaps with each other station individually first
            pairwise_overlaps = {}
            for other_codigo in overlapping_stations:
                other_circle = circles[other_codigo]['circle']
                try:
                    overlap = station_circle.intersection(other_circle)
                    if not overlap.is_empty and overlap.area > 1:
                        pairwise_overlaps[other_codigo] = overlap
                except Exception as e:
                    print(f"Warning: Error calculating pairwise overlap {codigo}-{other_codigo}: {e}")
                    continue
            
            # Handle 2-way overlaps (between this station and one other)
            processed_overlaps = set()
            
            for other_codigo in pairwise_overlaps:
                if other_codigo in processed_overlaps:
                    continue
                
                # Find all stations that overlap in this same region
                overlap_region = pairwise_overlaps[other_codigo]
                stations_in_region = [codigo, other_codigo]
                
                # Check if other stations also overlap in this same area
                for third_codigo in overlapping_stations:
                    if third_codigo != other_codigo and third_codigo not in processed_overlaps:
                        try:
                            third_circle = circles[third_codigo]['circle']
                            # Check if third station overlaps with both current stations in this region
                            if overlap_region.intersects(third_circle):
                                overlap_with_third = overlap_region.intersection(third_circle)
                                if not overlap_with_third.is_empty and overlap_with_third.area > 1:
                                    stations_in_region.append(third_codigo)
                                    overlap_region = overlap_with_third
                        except Exception as e:
                            print(f"Warning: Error checking 3-way overlap {codigo}-{other_codigo}-{third_codigo}: {e}")
                            continue
                
                # Now we have the final overlap region and all stations involved
                if len(stations_in_region) > 1:
                    try:
                        # This station gets 1/n of this overlap
                        station_share = overlap_region.area / len(stations_in_region)
                        shared_areas.append({
                            'area': station_share,
                            'geometry': overlap_region,
                            'shared_with': len(stations_in_region) - 1
                        })
                        
                        # Add to list of overlaps to subtract from exclusive area
                        all_overlaps_to_subtract.append(overlap_region)
                        
                        # Mark these stations as processed for this region
                        for station_in_region in stations_in_region[1:]:  # Skip current station
                            processed_overlaps.add(station_in_region)
                            
                    except Exception as e:
                        print(f"Warning: Error processing overlap region for station {codigo}: {e}")
                        continue
            
            # Remove all overlaps from exclusive area
            try:
                for overlap_to_subtract in all_overlaps_to_subtract:
                    exclusive_area = exclusive_area.difference(overlap_to_subtract)
                    
                # Handle potential geometry issues
                if exclusive_area.is_empty:
                    exclusive_area = station_circle.buffer(0)  # Try to fix geometry
                    
            except Exception as e:
                print(f"Warning: Error calculating exclusive area for station {codigo}: {e}")
                exclusive_area = station_circle  # Fallback to original circle
            
            # Calculate final area
            exclusive_area_size = exclusive_area.area if hasattr(exclusive_area, 'area') else 0
            shared_area_size = sum(shared['area'] for shared in shared_areas)
            final_area = exclusive_area_size + shared_area_size
            
            territories.append({
                'station_codigo': codigo,  # Use codigo_estacao
                'station_name': station_name,
                'df_index': idx,
                'territory': exclusive_area,
                'original_area': station_circle.area,
                'final_area': final_area,
                'exclusive_area': exclusive_area_size,
                'shared_areas': shared_areas
            })
            
        except Exception as e:
            print(f"Error processing station {codigo}: {e}")
            # Fallback: use original circle
            territories.append({
                'station_codigo': codigo,
                'station_name': station_name,
                'df_index': idx,
                'territory': station_circle,
                'original_area': station_circle.area,
                'final_area': station_circle.area,
                'exclusive_area': station_circle.area,
                'shared_areas': []
            })
    
    print(f"Processed {len(territories)} territories successfully.")
    return territories

# 9. Create non-overlapping territories
territories = create_non_overlapping_territories(points_projected)

# 10. Modified function to analyze territories with zones
def analyze_territory_zone_intersections(territories, zones_gdf, station_name_col=None):

    results = []
    
    for territory in territories:
        codigo = territory['station_codigo']  # Use codigo_estacao
        df_index = territory['df_index']
        exclusive_territory = territory['territory']
        shared_areas = territory.get('shared_areas', [])
        
        # Get station name using codigo_estacao
        station_name = f"STATION_{codigo}"
        
        station_result = {
            'station_name': station_name,
            'station_codigo': codigo,  # Use codigo_estacao
            'df_index': df_index,
            'original_circle_area_m2': territory['original_area'],
            'final_territory_area_m2': territory['final_area'],
            'exclusive_area_m2': territory['exclusive_area'],
            'shared_area_m2': sum(shared['area'] for shared in shared_areas),
            'intersections': []
        }
        
        # Find intersecting zones for exclusive territory
        try:
            if exclusive_territory.is_empty or not hasattr(exclusive_territory, 'area'):
                intersecting_zones = zones_gdf.iloc[0:0]  # Empty GeoDataFrame
            else:
                intersecting_zones = zones_gdf[zones_gdf.geometry.intersects(exclusive_territory)]
        except Exception as e:
            print(f"Warning: Error finding intersecting zones for station {codigo}: {e}")
            intersecting_zones = zones_gdf.iloc[0:0]  # Empty GeoDataFrame
        
        for zone_idx, zone_row in intersecting_zones.iterrows():
            zone_geom = zone_row.geometry
            zone_total_area = zone_row['total_area_m2']
            
            # Calculate intersection with exclusive territory
            try:
                intersection = exclusive_territory.intersection(zone_geom)
                intersection_area = intersection.area if not intersection.is_empty else 0
            except Exception as e:
                print(f"Warning: Error calculating intersection for station {codigo}, zone {zone_idx}: {e}")
                intersection_area = 0
            
            # Add proportional shares from overlapped areas
            shared_intersection_area = 0
            for shared in shared_areas:
                try:
                    shared_zone_intersection = shared['geometry'].intersection(zone_geom)
                    if not shared_zone_intersection.is_empty:
                        # This station gets its proportional share
                        shared_intersection_area += shared_zone_intersection.area / (shared['shared_with'] + 1)
                except Exception as e:
                    print(f"Warning: Error calculating shared intersection for station {codigo}: {e}")
                    continue
            
            total_intersection_area = intersection_area + shared_intersection_area
            
            if total_intersection_area > 1:  # Only include significant intersections
                # Calculate percentages based on final territory area
                pct_of_territory = (total_intersection_area / territory['final_area']) * 100
                pct_of_zone = (total_intersection_area / zone_total_area) * 100
                
                # Get zone identifier
                zone_cols = [col for col in zones_gdf.columns if 'name' in col.lower() or 'id' in col.lower() or 'zona' in col.lower()]
                if zone_cols:
                    zone_name = zone_row[zone_cols[0]]
                else:
                    zone_name = f"Zone_{zone_idx}"
                
                station_result['intersections'].append({
                    'zone_name': zone_name,
                    'intersection_area_m2': total_intersection_area,
                    'exclusive_intersection_m2': intersection_area,
                    'shared_intersection_m2': shared_intersection_area,
                    'pct_of_territory': pct_of_territory,
                    'pct_of_zone': pct_of_zone,
                    'zone_total_area_m2': zone_total_area
                })
        
        results.append(station_result)
    
    return results

# 11. Perform the analysis with overlap handling
analysis_results = analyze_territory_zone_intersections(
    territories, 
    zonas_projected, 
    station_name_col=None  # Change this to your station name column
)

# 12. Print results in the requested format
print("TERRITORY-ZONE INTERSECTION ANALYSIS (WITH OVERLAP HANDLING)")
print("=" * 60)

total_original_area = sum(t['original_area'] for t in territories)
total_final_area = sum(t['final_area'] for t in territories)
print(f"Total original circle area: {total_original_area:,.0f} m²")
print(f"Total final territory area: {total_final_area:,.0f} m²")
print(f"Overlap reduction: {total_original_area - total_final_area:,.0f} m²")
print("-" * 60)

for result in analysis_results:
    station_name = result['station_name']
    original_area = result['original_circle_area_m2']
    final_area = result['final_territory_area_m2']
    exclusive_area = result['exclusive_area_m2']
    shared_area = result['shared_area_m2']
    
    print(f"\n{station_name}:")
    print(f"  Original circle: {original_area:,.0f} m²")
    print(f"  Final territory: {final_area:,.0f} m² (exclusive: {exclusive_area:,.0f} m² + shared: {shared_area:,.0f} m²)")
    
    if not result['intersections']:
        print(f"  No zone intersections found")
        continue
    
    # Sort intersections by percentage of territory (descending)
    intersections = sorted(result['intersections'], key=lambda x: x['pct_of_territory'], reverse=True)
    
    intersection_parts = []
    for intersection in intersections:
        zone_name = intersection['zone_name']
        pct_territory = intersection['pct_of_territory']
        pct_zone = intersection['pct_of_zone']
        exclusive_int = intersection['exclusive_intersection_m2']
        shared_int = intersection['shared_intersection_m2']
        
        detail = f"{pct_territory:.1f}% in {zone_name} (comprising {pct_zone:.1f}% of total zone area)"
        if shared_int > 1:
            detail += f" [excl: {exclusive_int:,.0f}m², shared: {shared_int:,.0f}m²]"
        
        intersection_parts.append(detail)
    
    print(f"  Zones: {' + '.join(intersection_parts)}")

# 13. Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

# Original overlapping circles
ax1.set_title("Original Overlapping 400m Circles", fontsize=14)
zonas_projected.plot(ax=ax1, facecolor="#A0C8F0", edgecolor="black", alpha=0.7, linewidth=0.5)
points_projected.plot(ax=ax1, color='red', markersize=8, marker='o')

circles_gdf = gpd.GeoDataFrame(points_projected.drop('geometry', axis=1), 
                              geometry=points_projected['circle'])
circles_gdf.plot(ax=ax1, facecolor='none', edgecolor='red', alpha=0.8, 
                linewidth=1.5)

# Use codigo_estacao for annotations in first subplot
for idx, row in points_projected.iterrows():
    x, y = row.geometry.x, row.geometry.y
    codigo = row['codigo_estacao']  # Get codigo_estacao
    ax1.annotate(f'{codigo}', (x, y), xytext=(5, 5), textcoords='offset points', 
                fontsize=8, ha='left')

ax1.axis('off')

# Modified territories (exclusive areas only for visualization clarity)
ax2.set_title("Exclusive Territories (Overlaps Divided)", fontsize=14)
zonas_projected.plot(ax=ax2, facecolor="#A0C8F0", edgecolor="black", alpha=0.7, linewidth=0.5)
points_projected.plot(ax=ax2, color='red', markersize=8, marker='o')

# Plot exclusive territories
for territory in territories:
    try:
        if hasattr(territory['territory'], 'area') and not territory['territory'].is_empty:
            territory_gdf = gpd.GeoDataFrame([territory], geometry=[territory['territory']])
            territory_gdf.plot(ax=ax2, facecolor='none', edgecolor='blue', alpha=0.8, 
                              linewidth=1.5)
    except Exception as e:
        print(f"Warning: Could not plot territory for station {territory['station_codigo']}: {e}")
        continue

# Use codigo_estacao for annotations in second subplot
for idx, row in points_projected.iterrows():
    x, y = row.geometry.x, row.geometry.y
    codigo = row['codigo_estacao']  # Get codigo_estacao
    ax2.annotate(f'{codigo}', (x, y), xytext=(5, 5), textcoords='offset points', 
                fontsize=8, ha='left')

ax2.axis('off')
plt.tight_layout()
plt.show()

# Create CSV outputs with overlap handling
#detailed_df, summary_df = create_csv_output_with_overlaps(analysis_results, 'scripts/dados/porcentagem_zonas_no_overlap.csv')

# 15. NEW: Create clean Python data structure with station -> zones -> percentages
def create_station_zone_dict(analysis_results):
   
    station_zones = {}
    
    for result in analysis_results:
        codigo = result['station_codigo']  # Use codigo_estacao as key
        station_zones[codigo] = {}
        
        # Add all zone intersections with their percentages
        for intersection in result['intersections']:
            zone_name = intersection['zone_name']
            pct_of_zone = round(intersection['pct_of_zone'], 2)
            
            # Only include zones where the station comprises more than 0.01% (to filter tiny intersections)
            if pct_of_zone > 0.01:
                station_zones[codigo][zone_name] = pct_of_zone
    
    return station_zones

# Create the data structure
station_zone_data = create_station_zone_dict(analysis_results)

# 16. Display the results in a clean format
print("\n" + "="*80)
print("STATION -> ZONE PERCENTAGES DATA STRUCTURE")
print("="*80)
print("Format: {codigo_estacao: {zone_name: percentage_of_zone_comprised_by_station}}")
print("-"*80)

# Pretty print the dictionary
import json
for codigo, zones_dict in station_zone_data.items():
    if zones_dict:  # Only show stations that have zone intersections
        print(f"\n'{codigo}': {{")
        for zone_name, percentage in sorted(zones_dict.items(), key=lambda x: x[1], reverse=True):
            print(f"    '{zone_name}': {percentage}%,")
        print("}")
    else:
        print(f"\n'{codigo}': {{}} # No zone intersections")

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

# Calculate some summary statistics
total_stations = len(station_zone_data)
stations_with_zones = sum(1 for zones in station_zone_data.values() if zones)
total_zone_intersections = sum(len(zones) for zones in station_zone_data.values())

print(f"Total stations: {total_stations}")
print(f"Stations with zone intersections: {stations_with_zones}")
print(f"Total station-zone intersections: {total_zone_intersections}")
print(f"Average zones per station: {total_zone_intersections/total_stations:.1f}")

# Find zones that are most affected by stations
zone_impact = {}
for codigo, zones_dict in station_zone_data.items():
    for zone_name, percentage in zones_dict.items():
        if zone_name not in zone_impact:
            zone_impact[zone_name] = []
        zone_impact[zone_name].append((codigo, percentage))

print(f"\nZones affected by stations: {len(zone_impact)}")

# Show top affected zones
print("\nTOP 5 MOST AFFECTED ZONES:")
zone_totals = {}
for zone_name, stations in zone_impact.items():
    zone_totals[zone_name] = sum(pct for _, pct in stations)

top_zones = sorted(zone_totals.items(), key=lambda x: x[1], reverse=True)[:5]
for zone_name, total_pct in top_zones:
    stations_in_zone = zone_impact[zone_name]
    stations_str = ", ".join([f"{codigo}({pct}%)" for codigo, pct in sorted(stations_in_zone, key=lambda x: x[1], reverse=True)])
    print(f"  {zone_name}: {total_pct:.1f}% total coverage")
    print(f"    Stations: {stations_str}")

# 17. Export as clean JSON file
def export_station_zones_json(station_zone_data, filename='station_zones_1000.json'):
    """Export the clean data structure as JSON"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(station_zone_data, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Clean data structure exported to: {filename}")
    except Exception as e:
        print(f"Error exporting JSON: {e}")
    
    return station_zone_data

# Export the data structure
final_data = export_station_zones_json(station_zone_data, 'scripts/dados/station_zones_882.json')



print("Structure: {codigo_estacao: {zone_name: percentage_of_zone}}")
