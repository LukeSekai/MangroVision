"""Snapshot only the fields changed by the organization-account conversion.

This is an ownership/account rollback snapshot, not a full database backup.
No passwords, password hashes, session tokens, or session hashes are exported.
"""
import gzip
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqlalchemy import text
from mangrovision_db.compat import get_engine


def main():
    selections = {
        'planters': 'id, organization_id, full_name, status',
        'planter_assignments': 'id, planter_id',
        'planting_events': 'id, planter_id',
        'point_death_records': 'id, planter_id',
        'auth_sessions': 'id, revoked_at',
    }
    with get_engine().connect() as connection:
        connection.execute(text('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY'))
        revision = connection.execute(text('SELECT version_num FROM mangrovision.alembic_version')).scalar_one()
        if revision != '20260912_0005':
            raise RuntimeError('This snapshot requires the pre-conversion revision 20260912_0005.')
        snapshot = {'revision': revision, 'created_at': datetime.now(timezone.utc).isoformat(), 'tables': {}}
        for table, columns in selections.items():
            where = " WHERE subject_type = 'planter'" if table == 'auth_sessions' else ''
            rows = connection.execute(text(f'SELECT {columns} FROM mangrovision.{table}{where} ORDER BY id')).mappings()
            snapshot['tables'][table] = [dict(row) for row in rows]
        snapshot['organization_planting_totals'] = [dict(row) for row in connection.execute(text('''
            SELECT p.organization_id, COUNT(*) AS total FROM mangrovision.planting_events pe
            JOIN mangrovision.planters p ON p.id = pe.planter_id GROUP BY p.organization_id
            ORDER BY p.organization_id''')).mappings()]
    directory = ROOT / 'backups'
    directory.mkdir(exist_ok=True)
    path = directory / f"organization-accounts-before-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json.gz"
    payload = json.dumps(snapshot, default=str, indent=2).encode()
    with gzip.open(path, 'xb') as output:
        output.write(payload)
    with gzip.open(path, 'rb') as check:
        assert check.read() == payload
    print('Conversion snapshot:', path)
    print('SHA256:', hashlib.sha256(payload).hexdigest())
    print('Saved row counts:', {table: len(rows) for table, rows in snapshot['tables'].items()})


if __name__ == '__main__':
    main()
