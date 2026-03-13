"""Quick test of the updated 5-table schema."""
import sys, sqlite3
sys.path.insert(0, '.')
from planting_database import *

DB = "planting_zones.db"
conn = sqlite3.connect(DB)

# 1. Tables
tables = [r[0] for r in conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
).fetchall()]
print("Tables:", tables)
assert "login_history" in tables, "login_history table missing!"

# 2. Indices
indices = [r[0] for r in conn.execute(
    "SELECT name FROM sqlite_master WHERE type='index' ORDER BY name"
).fetchall()]
print("Indices:", indices)
assert "idx_login_user" in indices
assert "idx_login_date" in indices

# 3. login_history columns
cols = [r[1] for r in conn.execute("PRAGMA table_info(login_history)").fetchall()]
print("login_history cols:", cols)
assert "logged_in_at" in cols

# 4. analyses should NOT have detection_mode
a_cols = [r[1] for r in conn.execute("PRAGMA table_info(analyses)").fetchall()]
print("analyses cols:", a_cols)
assert "detection_mode" not in a_cols, "detection_mode still in analyses!"
print("  CONFIRMED: detection_mode removed")

# 5. users should NOT have last_login (replaced by login_history table)
u_cols = [r[1] for r in conn.execute("PRAGMA table_info(users)").fetchall()]
print("users cols:", u_cols)
assert "last_login" not in u_cols, "last_login still in users!"
print("  CONFIRMED: last_login moved to login_history table")

# 6. Test login recording
uid = get_or_create_default_user()
lid1 = record_login(uid, ip_address="127.0.0.1", user_agent="MangroVision/1.0")
lid2 = record_login(uid, ip_address="192.168.1.10", user_agent="Chrome/120")
print(f"Recorded logins: id={lid1}, id={lid2}")

history = get_login_history(uid)
print(f"Login history: {len(history)} entries")
for h in history:
    print(f"  [{h['logged_in_at']}] user={h['full_name']} ip={h['ip_address']}")
assert len(history) == 2, f"Expected 2 logins, got {len(history)}"

# 7. Test save_analysis still works without detection_mode
results = {
    "total_area_m2": 500,
    "canopy_count": 12,
    "polygon_count": 8,
    "danger_area_m2": 50,
    "danger_percentage": 10,
    "plantable_area_m2": 200,
    "plantable_percentage": 40,
    "ai_confidence": 0.85,
}
hexagons = [
    {"_gps_lat": 10.5001, "_gps_lon": 122.5001, "center": (100, 200), "area_m2": 25},
]
aid, new_pts, skipped = save_analysis("test.jpg", 10.5, 122.5, results, hexagons)
print(f"Saved analysis id={aid}, points={new_pts}, skipped={skipped}")
assert new_pts == 1

conn.close()
print("\n✅ ALL 5-TABLE SCHEMA TESTS PASSED!")
