"""Full integration test: simulate what app.py does with the database."""
import sys, os
sys.path.insert(0, r'c:\Users\Lenovo-Pc\OneDrive\Thesis\thesis_env\MangroVision')
os.chdir(r'c:\Users\Lenovo-Pc\OneDrive\Thesis\thesis_env\MangroVision')

from planting_database import (
    save_analysis, find_overlapping_analyses, count_nearby_points,
    get_all_stats, get_all_planting_points, delete_analysis,
)

# ── Simulate what app.py does after analysis ──────────────────────
fake_results = {
    'altitude_m': 6.0,
    'gsd_m_per_pixel': 0.0022,
    'coverage_m': [8.4, 4.7],
    'total_area_m2': 39.5,
    'canopy_count': 15,
    'polygon_count': 3,
    'danger_area_m2': 39.4,
    'danger_percentage': 99.4,
    'plantable_area_m2': 0.25,
    'plantable_percentage': 0.6,
    'hexagon_count': 0,
    '_forbidden_filtered': 0,
    '_eroded_filtered': 0,
}

fake_hexagons = [
    {'_gps_lat': 10.780001, '_gps_lon': 122.625001, 'center': (100, 200), 'buffer_radius_m': 1.0, 'area_m2': 0.86},
    {'_gps_lat': 10.780002, '_gps_lon': 122.625002, 'center': (150, 250), 'buffer_radius_m': 1.0, 'area_m2': 0.86},
    {'_gps_lat': 10.780003, '_gps_lon': 122.625003, 'center': (200, 300), 'buffer_radius_m': 1.0, 'area_m2': 0.86},
]

# 1. save_analysis (exact same call as app.py line 1755)
aid, new_count, skipped = save_analysis(
    image_name='mangrove.jpg',
    center_lat=10.780000,
    center_lon=122.625000,
    results=fake_results,
    hexagons=fake_hexagons,
)
print(f"1. save_analysis: id={aid}, new={new_count}, skipped={skipped}")
assert new_count == 3

# 2. find_overlapping_analyses (app.py line 976)
overlaps = find_overlapping_analyses(10.780000, 122.625000)
print(f"2. find_overlapping: {len(overlaps)} found")
assert len(overlaps) == 1
assert overlaps[0]['image_name'] == 'mangrove.jpg'

# 3. count_nearby_points (app.py line 978)
nearby = count_nearby_points(10.780000, 122.625000)
print(f"3. count_nearby_points: {nearby}")
assert nearby == 3

# 4. get_all_stats (app.py line 543)
stats = get_all_stats()
print(f"4. get_all_stats: {stats['total_analyses']} analyses, {stats['total_planting_points']} points")
assert stats['total_analyses'] == 1
assert stats['total_planting_points'] == 3
assert len(stats['analyses']) == 1
assert len(stats['points']) == 3
# Verify backward-compatible keys that app.py reads
a = stats['analyses'][0]
for key in ['id', 'image_name', 'analyzed_at', 'center_lat', 'center_lon',
            'canopy_count', 'hexagon_count', 'plantable_area_m2',
            'danger_area_m2', 'total_area_m2', 'gsd_cm',
            'forbidden_filtered', 'eroded_filtered']:
    assert key in a, f"Missing key in analysis: {key}"
print(f"   All app.py-required keys present in stats")

# 5. get_all_planting_points (app.py line 42)
pts = get_all_planting_points()
print(f"5. get_all_planting_points: {len(pts)}")
assert len(pts) == 3

# 6. Re-run save_analysis for same area → should REPLACE
aid2, new2, skip2 = save_analysis(
    image_name='mangrove_v2.jpg',
    center_lat=10.780001,
    center_lon=122.625001,
    results=fake_results,
    hexagons=fake_hexagons,
)
stats2 = get_all_stats()
print(f"6. Re-analysis replace: old analysis replaced, now {stats2['total_analyses']} analysis(es)")
assert stats2['total_analyses'] == 1  # replaced, not duplicated

# 7. delete_analysis (app.py line 695)
delete_analysis(aid2)
stats3 = get_all_stats()
print(f"7. delete_analysis: {stats3['total_analyses']} analyses remaining")
assert stats3['total_analyses'] == 0

# Verify CASCADE: planting points should be gone too
pts3 = get_all_planting_points()
assert len(pts3) == 0
print(f"   CASCADE delete verified: 0 orphan points")

print("\n✅ Full integration test passed! System is fully compatible.")
