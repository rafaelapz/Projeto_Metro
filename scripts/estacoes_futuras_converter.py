

import pandas as pd
import re
import time
from typing import Tuple, Optional

# Try to import optional libraries
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("Warning: requests not installed. Install with: pip install requests")

try:
    import googlemaps
    HAS_GOOGLEMAPS = True
except ImportError:
    HAS_GOOGLEMAPS = False
    print("Note: googlemaps not installed. Google Maps geocoding will not be available.")

try:
    from apikeys import API_KEY_GOOGLE
    HAS_API_KEY = True
except ImportError:
    HAS_API_KEY = False
    print("Note: Could not import API_KEY_GOOGLE from apikeys.py")


def clean_coordinate_string(coord_str: str) -> str:
    """Clean and fix common coordinate string issues"""
    if not coord_str:
        return coord_str
    
    # Fix common typo: missing leading digit (e.g., "3°" should be "23°")
    coord_str = re.sub(r'^(\d)°', r'2\1°', coord_str)
    
    # Replace various quote marks with standard ones
    coord_str = coord_str.replace('"', '"').replace('"', '"').replace("''", '"')
    coord_str = coord_str.replace("'", "'").replace("'", "'").replace("′", "'").replace("″", '"')
    
    # Remove any extra text (like "Connects to Line 20")
    coord_str = re.sub(r'\.?\s*Connects.*', '', coord_str)
    
    return coord_str.strip()


def dms_to_decimal(dms_str: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Convert DMS (Degrees Minutes Seconds) format to decimal degrees
    Handles various formats and common errors
    """
    try:
        # Clean the string
        dms_str = clean_coordinate_string(str(dms_str).strip())
        
        if not dms_str or dms_str.lower() == 'nan':
            return None, None
        
        # Split by various possible separators
        parts = None
        for separator in [' ', ', ', '; ', '  ', ' - ', '. ']:
            test_parts = dms_str.split(separator)
            # Look for two parts that both contain degree symbols
            if len(test_parts) >= 2:
                if '°' in test_parts[0] and '°' in test_parts[1]:
                    parts = [test_parts[0], test_parts[1]]
                    break
        
        if not parts or len(parts) != 2:
            print(f"    Could not split coordinates: {dms_str}")
            return None, None
        
        lat_dms = parts[0].strip()
        lon_dms = parts[1].strip()
        
        def parse_dms(dms: str) -> Optional[float]:
            """Parse individual DMS string"""
            patterns = [
                r"(\d+)°(\d+)'([\d.]+)[\"']?([NSEW])",  # Standard format
                r"(\d+)°(\d+)'([\d.]+)\s*([NSEW])",      # Space before direction
                r"(\d+)°\s*(\d+)'\s*([\d.]+)[\"']?\s*([NSEW])",  # Spaces
                r"(\d+)°(\d+)['′]([\d.]+)[\"″]?([NSEW])",  # Unicode symbols
            ]
            
            for pattern in patterns:
                match = re.match(pattern, dms.strip())
                if match:
                    degrees = float(match.group(1))
                    minutes = float(match.group(2))
                    seconds = float(match.group(3))
                    direction = match.group(4)
                    
                    # Convert to decimal
                    decimal = degrees + minutes/60 + seconds/3600
                    
                    # Apply sign based on direction
                    if direction in ['S', 'W']:
                        decimal = -decimal
                    
                    return round(decimal, 6)
            
            return None
        
        latitude = parse_dms(lat_dms)
        longitude = parse_dms(lon_dms)
        
        # Validate São Paulo coordinates (roughly)
        if latitude and longitude:
            if not (-24.0 < latitude < -23.0):
                print(f"    Warning: Latitude {latitude} seems outside São Paulo region")
            if not (-47.0 < longitude < -46.0):
                print(f"    Warning: Longitude {longitude} seems outside São Paulo region")
        
        return latitude, longitude
        
    except Exception as e:
        print(f"    Error parsing DMS: {e}")
        return None, None


def geocode_nominatim(address: str) -> Tuple[Optional[float], Optional[float]]:
    """Geocode using OpenStreetMap Nominatim (free, no API key required)"""
    if not HAS_REQUESTS:
        return None, None
    
    try:
        import requests
        
        # Respect Nominatim's usage policy
        time.sleep(1)
        
        # Add context for better results
        full_address = f"{address}, São Paulo, Brazil"
        
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            'q': full_address,
            'format': 'json',
            'limit': 1,
            'countrycodes': 'br'  # Restrict to Brazil
        }
        headers = {
            'User-Agent': 'Station Geocoder Script 1.0'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0:
                lat = float(data[0]['lat'])
                lon = float(data[0]['lon'])
                return round(lat, 6), round(lon, 6)
        
        return None, None
        
    except Exception as e:
        print(f"    Nominatim error: {e}")
        return None, None


def geocode_google(address: str, gmaps) -> Tuple[Optional[float], Optional[float]]:
    """Geocode using Google Maps API"""
    if not gmaps:
        return None, None
    
    try:
        full_address = f"{address}, São Paulo, Brazil"
        result = gmaps.geocode(full_address)
        
        if result and len(result) > 0:
            location = result[0]['geometry']['location']
            return round(location['lat'], 6), round(location['lng'], 6)
        
        return None, None
        
    except Exception as e:
        if "REQUEST_DENIED" in str(e):
            print(f"    Google Maps API requires billing to be enabled")
        else:
            print(f"    Google Maps error: {e}")
        return None, None


def search_google_by_coordinate(lat: float, lon: float, gmaps) -> str:
    """Reverse geocode to get address from coordinates"""
    if not gmaps:
        return ""
    
    try:
        result = gmaps.reverse_geocode((lat, lon))
        if result and len(result) > 0:
            return result[0].get('formatted_address', '')
        return ""
    except:
        return ""


def process_stations(input_file: str, output_file: str):
    """Main processing function"""
    
    # Read Excel file
    print(f"\n{'='*70}")
    print("LOADING EXCEL FILE")
    print(f"{'='*70}")
    
    df = pd.read_excel(input_file)
    print(f"Loaded {len(df)} stations")
    print(f"Columns: {df.columns.tolist()}")
    
    # Verify we have the expected columns
    expected_cols = ['codigo_estacao', 'nome', 'address', 'coordinates']
    actual_cols = df.columns.tolist()
    
    # Map columns if names are slightly different
    column_mapping = {}
    for expected in expected_cols:
        for actual in actual_cols:
            if expected in actual.lower() or actual.lower() in expected:
                column_mapping[expected] = actual
                break
    
    # Use mapped column names
    codigo_col = column_mapping.get('codigo_estacao', 'codigo_estacao')
    nome_col = column_mapping.get('nome', 'nome')
    address_col = column_mapping.get('address', 'address')
    coords_col = column_mapping.get('coordinates', 'coordinates')
    
    print(f"\nColumn mapping:")
    print(f"  codigo_estacao -> {codigo_col}")
    print(f"  nome -> {nome_col}")
    print(f"  address -> {address_col}")
    print(f"  coordinates -> {coords_col}")
    
    # Initialize Google Maps client if available
    gmaps = None
    if HAS_GOOGLEMAPS and HAS_API_KEY:
        try:
            gmaps = googlemaps.Client(key=API_KEY_GOOGLE)
            print("\n✓ Google Maps client initialized (may fail if billing not enabled)")
        except Exception as e:
            print(f"\n✗ Could not initialize Google Maps: {e}")
    
    # Process each station
    print(f"\n{'='*70}")
    print("PROCESSING STATIONS")
    print(f"{'='*70}")
    
    results = []
    stats = {
        'coords_success': 0,
        'coords_failed': 0,
        'geocode_success': 0,
        'geocode_failed': 0,
        'no_data': 0
    }
    
    for idx, row in df.iterrows():
        codigo = row[codigo_col] if codigo_col in row else idx
        nome = row[nome_col] if nome_col in row else ''
        address = str(row[address_col]) if address_col in row and pd.notna(row[address_col]) else ''
        coordinates = str(row[coords_col]) if coords_col in row and pd.notna(row[coords_col]) else ''
        
        # Clean up
        address = address.strip() if address.lower() != 'nan' else ''
        coordinates = coordinates.strip() if coordinates.lower() != 'nan' else ''
        
        print(f"\n[{codigo}] {nome}")
        
        latitude = None
        longitude = None
        source = ''
        
        # Priority 1: Try coordinates first (no API needed)
        if coordinates and '°' in coordinates:
            print(f"  DMS: {coordinates[:50]}...")
            latitude, longitude = dms_to_decimal(coordinates)
            
            if latitude is not None and longitude is not None:
                print(f"  ✓ Converted: {latitude}, {longitude}")
                source = 'DMS coordinates'
                stats['coords_success'] += 1
                
                # Optionally reverse geocode to get address
                if gmaps and not address:
                    found_address = search_google_by_coordinate(latitude, longitude, gmaps)
                    if found_address:
                        address = found_address
                        print(f"  Found address: {address[:50]}...")
            else:
                print(f"  ✗ Failed to parse DMS")
                stats['coords_failed'] += 1
        
        # Priority 2: Try address if no coordinates or coordinate parsing failed
        if latitude is None and address:
            print(f"  Address: {address[:50]}...")
            
            # Try Google Maps first if available
            if gmaps:
                latitude, longitude = geocode_google(address, gmaps)
                if latitude is not None:
                    source = 'Google Maps geocoding'
            
            # Fallback to Nominatim
            if latitude is None and HAS_REQUESTS:
                print(f"  Trying OpenStreetMap...")
                latitude, longitude = geocode_nominatim(address)
                if latitude is not None:
                    source = 'OpenStreetMap geocoding'
            
            if latitude is not None:
                print(f"  ✓ Geocoded: {latitude}, {longitude}")
                stats['geocode_success'] += 1
            else:
                print(f"  ✗ Geocoding failed")
                stats['geocode_failed'] += 1
        
        # No data available
        if latitude is None:
            if not coordinates and not address:
                print(f"  → No data available")
                stats['no_data'] += 1
                source = 'No data'
        
        # Store result
        results.append({
            'codigo_estacao': codigo,
            'nome': nome,
            'latitude': latitude,
            'longitude': longitude,
            'address': address if address else '',
            'original_coordinates': coordinates if coordinates else '',
            'source': source
        })
    
    # Create DataFrame and save
    result_df = pd.DataFrame(results)
    
    # Save main output
    result_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved all results to: {output_file}")
    
    # Save successful conversions only (simplified)
    successful = result_df[result_df['latitude'].notna()].copy()
    if len(successful) > 0:
        success_file = output_file.replace('.csv', '_successful.csv')
        successful[['codigo_estacao', 'nome', 'latitude', 'longitude']].to_csv(success_file, index=False)
        print(f"✓ Successful conversions: {success_file}")
    
    # Save failed conversions for review
    failed = result_df[result_df['latitude'].isna()].copy()
    if len(failed) > 0:
        failed_file = output_file.replace('.csv', '_failed.csv')
        failed.to_csv(failed_file, index=False)
        print(f"✓ Failed conversions: {failed_file}")
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    total = len(result_df)
    successful_count = result_df['latitude'].notna().sum()
    failed_count = total - successful_count
    
    print(f"Total stations: {total}")
    print(f"Successfully processed: {successful_count} ({100*successful_count/total:.1f}%)")
    print(f"Failed: {failed_count} ({100*failed_count/total:.1f}%)")
    
    print(f"\nBreakdown:")
    print(f"  Coordinates (DMS):")
    print(f"    ✓ Parsed successfully: {stats['coords_success']}")
    print(f"    ✗ Failed to parse: {stats['coords_failed']}")
    print(f"  Address geocoding:")
    print(f"    ✓ Geocoded successfully: {stats['geocode_success']}")
    print(f"    ✗ Failed to geocode: {stats['geocode_failed']}")
    print(f"  No data provided: {stats['no_data']}")
    
    # Show sample results
    if successful_count > 0:
        print(f"\n{'='*70}")
        print("SAMPLE RESULTS (first 10 successful)")
        print(f"{'='*70}")
        sample = successful[['codigo_estacao', 'nome', 'latitude', 'longitude']].head(10)
        print(sample.to_string(index=False))
    
    # List failed stations
    if failed_count > 0 and failed_count <= 20:
        print(f"\n{'='*70}")
        print("FAILED STATIONS")
        print(f"{'='*70}")
        for _, row in failed.iterrows():
            print(f"[{row['codigo_estacao']}] {row['nome']}")
            if row['original_coordinates']:
                print(f"  Coordinates: {row['original_coordinates'][:50]}...")
            if row['address']:
                print(f"  Address: {row['address'][:50]}...")
    
    return result_df


def main():
    """Main entry point"""
    input_file = "/Users/ellazyngier/Documents/github/tccII/scripts/dados/estacoes_futuras.xlsx"
    output_file = "/Users/ellazyngier/Documents/github/tccII/scripts/dados/estacoes_futuras_processed.csv"
    
    print("\n" + "="*70)
    print("STATION DATA PROCESSOR")
    print("="*70)
    print("This script will:")
    print("1. Convert DMS coordinates to decimal degrees")
    print("2. Geocode addresses using OpenStreetMap (free)")
    print("3. Use Google Maps if available (requires billing)")
    
    # Check dependencies
    print("\nDependencies:")
    print(f"  pandas: ✓")
    print(f"  requests: {'✓' if HAS_REQUESTS else '✗ (install with: pip install requests)'}")
    print(f"  googlemaps: {'✓' if HAS_GOOGLEMAPS else '✗ (optional, install with: pip install googlemaps)'}")
    
    if not HAS_REQUESTS:
        print("\nInstalling requests for OpenStreetMap geocoding...")
        import subprocess
        subprocess.run(["pip", "install", "requests"])
        print("Please run the script again after installation.")
        return
    
    # Process the file
    result_df = process_stations(input_file, output_file)
    
    print("\n" + "="*70)
    print("PROCESSING COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()