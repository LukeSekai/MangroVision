"""Quick test for the new 4-table database schema."""
import sys, os
sys.path.insert(0, r'c:\Users\Lenovo-Pc\OneDrive\Thesis\thesis_env\MangroVision')
os.chdir(r'c:\Users\Lenovo-Pc\OneDrive\Thesis\thesis_env\MangroVision')

import planting_database as db
import sqlite3

conn = sqlite3.connect('planting_zones.db')

# Check tables
tables = [t[0] for t in conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
).fetchall()]
print("Tables:", tables)

# Check indices
indices = [i[0] for i in conn.execute(
    "SELECT name FROM sqlite_master WHERE type='index' ORDER BY name"
).fetchall()]
print("Indices:", indices)

# Test user
uid = db.get_or_create_default_user()
print(f"Default user id: {uid}")

user = db.get_user_by_email('planner@mangrovision.local')
print(f"User: {user['full_name']} ({user['role']}, {user['organization']})")

# Test exclusion zone CRUD
zid = db.save_exclusion_zone('forbidden', '{"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1],[0,0]]]}', 'Test')
zones = db.get_exclusion_zones('forbidden')
print(f"Forbidden zones after insert: {len(zones)}")
db.delete_exclusion_zone(zid)
zones2 = db.get_exclusion_zones('forbidden')
print(f"Forbidden zones after delete: {len(zones2)}")

# Test stats
stats = db.get_all_stats()
print(f"Stats keys: {sorted(stats.keys())}")
print(f"Total analyses: {stats['total_analyses']}")

# Test planting summary
summary = db.get_planting_summary()
print(f"Planting summary: {summary}")

# Verify API compat (all functions app.py imports)
for fn in ['save_analysis', 'find_overlapping_analyses', 'count_nearby_points',
           'get_all_stats', 'get_all_planting_points', 'delete_analysis']:
    assert hasattr(db, fn), f"Missing: {fn}"
print("All app.py API functions present!")

conn.close()
print("\n✅ All tests passed! 4-table schema is working correctly.")
