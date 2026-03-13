import sys, sqlite3
sys.path.insert(0, '.')
from planting_database import *

conn = sqlite3.connect('planting_zones.db')
tables = sorted(r[0] for r in conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence'"
).fetchall())
print('Tables:', tables)
assert tables == ['analyses', 'exclusion_zones', 'planting_points', 'users'], f'Unexpected: {tables}'
assert 'login_history' not in tables, 'login_history should not exist!'

u_cols = [r[1] for r in conn.execute('PRAGMA table_info(users)').fetchall()]
print('users cols:', u_cols)
assert 'last_login' in u_cols, 'last_login missing from users!'

a_cols = [r[1] for r in conn.execute('PRAGMA table_info(analyses)').fetchall()]
assert 'detection_mode' not in a_cols, 'detection_mode should be gone!'

uid = get_or_create_default_user()
update_last_login(uid)
row = conn.execute('SELECT last_login FROM users WHERE id = ?', (uid,)).fetchone()
print(f'last_login = {row[0]}')
assert row[0] is not None

conn.close()
print('ALL PASSED - 4 tables, last_login on users, no login_history, no detection_mode')
