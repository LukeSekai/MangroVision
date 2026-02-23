"""
MangroVision - Test Forbidden Zone Filter
Test script to verify forbidden zone filtering is working correctly
"""

import sys
from pathlib import Path

# Add canopy_detection to path
sys.path.append(str(Path(__file__).parent / "canopy_detection"))

from canopy_detection.forbidden_zone_filter import ForbiddenZoneFilter


def test_forbidden_zones():
    """Test the forbidden zone filter with sample coordinates"""
    
    print("=" * 70)
    print("MangroVision - Forbidden Zone Filter Test")
    print("=" * 70)
    print()
    
    # Initialize filter
    geojson_path = Path(__file__).parent / 'forbidden_zones.geojson'
    
    if not geojson_path.exists():
        print(f"❌ ERROR: forbidden_zones.geojson not found!")
        print(f"   Expected location: {geojson_path}")
        print()
        print("Please create the forbidden zones file using QGIS (Phase 1) first.")
        return False
    
    print(f"📂 Loading forbidden zones from: {geojson_path.name}")
    filter_obj = ForbiddenZoneFilter(str(geojson_path))
    print()
    
    # Get statistics
    stats = filter_obj.get_statistics()
    print("📊 Forbidden Zone Statistics:")
    print(f"   • Zones loaded: {stats['zone_count']}")
    print(f"   • File loaded: {stats['file_loaded']}")
    print()
    
    if stats['zone_count'] == 0:
        print("⚠️  No forbidden zones found in GeoJSON file!")
        print("   Please add polygons in QGIS and export again.")
        return False
    
    # Test coordinates
    # IMPORTANT: Replace these with actual coordinates from your area
    # These are example coordinates - you should use coordinates that:
    # 1. Are definitely INSIDE a forbidden zone
    # 2. Are definitely OUTSIDE any forbidden zone
    
    print("🧪 Testing Sample Coordinates:")
    print("   (Replace these with your actual area coordinates)")
    print()
    
    test_gaps = [
        {'lat': 10.750000, 'lon': 122.560000, 'name': 'Test Point 1'},
        {'lat': 10.750100, 'lon': 122.560100, 'name': 'Test Point 2'},
        {'lat': 10.750200, 'lon': 122.560200, 'name': 'Test Point 3'},
        {'lat': 10.750300, 'lon': 122.560300, 'name': 'Test Point 4'},
        {'lat': 10.750400, 'lon': 122.560400, 'name': 'Test Point 5'},
    ]
    
    print(f"Testing {len(test_gaps)} coordinates:")
    print()
    
    # Test individual points
    print("Individual Point Tests:")
    print("-" * 70)
    
    for gap in test_gaps:
        is_safe = filter_obj.is_safe_location(gap['lat'], gap['lon'])
        status = "✅ SAFE" if is_safe else "🚫 FORBIDDEN"
        print(f"{status} - {gap['name']}: ({gap['lat']:.6f}, {gap['lon']:.6f})")
    
    print()
    
    # Test batch filtering
    print("Batch Filtering Test:")
    print("-" * 70)
    
    safe_gaps, forbidden_gaps = filter_obj.filter_gaps(test_gaps)
    
    print(f"Total points tested: {len(test_gaps)}")
    print(f"✅ Safe points: {len(safe_gaps)}")
    print(f"🚫 Forbidden points: {len(forbidden_gaps)}")
    print()
    
    if len(forbidden_gaps) > 0:
        print("Filtered out locations:")
        for gap in forbidden_gaps:
            print(f"   • {gap['name']}: ({gap['lat']:.6f}, {gap['lon']:.6f})")
        print()
    
    print("=" * 70)
    print("Test Complete!")
    print("=" * 70)
    print()
    print("📝 Next Steps:")
    print("   1. Update test_gaps in this script with YOUR actual area coordinates")
    print("   2. Add coordinates that you KNOW are inside forbidden zones")
    print("   3. Add coordinates that you KNOW are outside forbidden zones")
    print("   4. Re-run this script to verify: python test_forbidden_zones.py")
    print()
    print("   When filtering works correctly:")
    print("   • Points on bridges/buildings should show 🚫 FORBIDDEN")
    print("   • Points in open mangrove areas should show ✅ SAFE")
    print()
    
    return True


def interactive_test():
    """Allow user to test individual coordinates interactively"""
    
    print()
    print("=" * 70)
    print("Interactive Testing Mode")
    print("=" * 70)
    print()
    
    geojson_path = Path(__file__).parent / 'forbidden_zones.geojson'
    
    if not geojson_path.exists():
        print(f"❌ ERROR: forbidden_zones.geojson not found!")
        return
    
    filter_obj = ForbiddenZoneFilter(str(geojson_path))
    
    if filter_obj.zone_count == 0:
        print("⚠️  No forbidden zones loaded!")
        return
    
    print(f"✅ Loaded {filter_obj.zone_count} forbidden zones")
    print()
    print("Enter coordinates to test (or 'quit' to exit):")
    print()
    
    while True:
        try:
            lat_input = input("Enter latitude (or 'quit'): ").strip()
            
            if lat_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Exiting...")
                break
            
            lon_input = input("Enter longitude: ").strip()
            
            try:
                lat = float(lat_input)
                lon = float(lon_input)
                
                is_safe = filter_obj.is_safe_location(lat, lon)
                
                if is_safe:
                    print(f"   ✅ SAFE - ({lat:.6f}, {lon:.6f}) is outside forbidden zones")
                else:
                    print(f"   🚫 FORBIDDEN - ({lat:.6f}, {lon:.6f}) is inside a forbidden zone")
                
                print()
                
            except ValueError:
                print("   ❌ Invalid coordinates! Please enter numbers.")
                print()
                
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break
        except EOFError:
            print("\n👋 Exiting...")
            break


if __name__ == "__main__":
    print()
    
    # Run automated tests
    success = test_forbidden_zones()
    
    if success:
        # Offer interactive mode
        response = input("\n🔍 Would you like to test individual coordinates? (y/n): ").strip().lower()
        
        if response in ['y', 'yes']:
            interactive_test()
    
    print()
    print("✨ Testing complete!")
    print()
