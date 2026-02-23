"""
Orthophoto Configuration Tool
Run this script to extract metadata from your QGIS/WebODM orthophoto
"""

from osgeo import gdal
import json
from pathlib import Path

def extract_orthophoto_metadata(orthophoto_path: str):
    """
    Extract geographic metadata from your orthophoto (GeoTIFF)
    
    Usage:
        1. Export your orthophoto from QGIS as GeoTIFF
        2. Run: python extract_orthophoto_info.py path/to/orthophoto.tif
        3. Copy the output values to map_backend.py OrthophotoMetadata class
    """
    
    # Open the raster
    dataset = gdal.Open(orthophoto_path)
    
    if dataset is None:
        print(f"❌ Could not open {orthophoto_path}")
        print("Make sure you have GDAL installed: pip install gdal")
        return
    
    # Get dimensions
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    
    # Get geotransform
    geotransform = dataset.GetGeoTransform()
    
    # Calculate bounds
    min_x = geotransform[0]
    max_y = geotransform[3]
    max_x = min_x + (width * geotransform[1])
    min_y = max_y + (height * geotransform[5])
    
    # Get CRS
    projection = dataset.GetProjection()
    
    # Calculate GSD (Ground Sample Distance)
    gsd_x = abs(geotransform[1])  # pixel width in map units
    gsd_y = abs(geotransform[5])  # pixel height in map units
    
    # Assume units are meters (typical for UTM)
    gsd_cm = (gsd_x * 100)  # Convert to centimeters
    
    print("\n" + "="*60)
    print("📍 ORTHOPHOTO METADATA EXTRACTED")
    print("="*60)
    
    print(f"\n📐 Image Dimensions:")
    print(f"   Width:  {width} pixels")
    print(f"   Height: {height} pixels")
    
    print(f"\n🌍 Geographic Bounds:")
    print(f"   North: {max_y}")
    print(f"   South: {min_y}")
    print(f"   East:  {max_x}")
    print(f"   West:  {min_x}")
    
    print(f"\n📏 Ground Sample Distance:")
    print(f"   GSD: {gsd_cm:.2f} cm/pixel")
    
    print(f"\n🗺️  Coordinate System:")
    print(f"   {projection[:100]}...")
    
    print("\n" + "="*60)
    print("📋 COPY THIS TO map_backend.py:")
    print("="*60)
    
    config = f"""
class OrthophotoMetadata:
    def __init__(self):
        # Geographic bounds (in lat/lon if WGS84, or projected coordinates)
        self.bounds = {{
            'north': {max_y},
            'south': {min_y},
            'east': {max_x},
            'west': {min_x}
        }}
        
        # Orthophoto dimensions
        self.width_px = {width}
        self.height_px = {height}
        
        # Coordinate system
        self.crs_from = "EPSG:4326"  # UPDATE if using UTM!
        self.crs_to = "EPSG:4326"
        
        # Ground Sample Distance
        self.gsd_cm = {gsd_cm:.2f}
"""
    
    print(config)
    print("="*60)
    
    # Save to JSON file
    metadata = {
        "bounds": {
            "north": float(max_y),
            "south": float(min_y),
            "east": float(max_x),
            "west": float(min_x)
        },
        "dimensions": {
            "width_px": width,
            "height_px": height
        },
        "gsd_cm": float(gsd_cm),
        "projection": projection
    }
    
    output_file = Path(orthophoto_path).stem + "_metadata.json"
    with open(output_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n💾 Metadata saved to: {output_file}")
    
    dataset = None  # Close dataset


if __name__ == "__main__":
    import sys
    
    print("🌿 MangroVision - Orthophoto Metadata Extractor\n")
    
    if len(sys.argv) < 2:
        print("Usage: python extract_orthophoto_info.py <path_to_orthophoto.tif>")
        print("\nExample:")
        print("  python extract_orthophoto_info.py orthophoto.tif")
        print("\nIf you don't have GDAL:")
        print("  pip install gdal")
        print("  or")
        print("  conda install gdal")
    else:
        orthophoto_path = sys.argv[1]
        extract_orthophoto_metadata(orthophoto_path)
