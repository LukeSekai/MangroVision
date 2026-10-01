"""Exercise the numbering migration transactionally, or verify the applied result."""

import argparse
import importlib.util
import sys
from pathlib import Path

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mangrovision_db.config import get_settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-applied', action='store_true')
    args = parser.parse_args()
    settings = get_settings()
    engine = create_engine(settings.migration_database_url, connect_args={'sslmode': settings.db_sslmode})
    with engine.connect() as conn:
        transaction = conn.begin()
        try:
            before = list(conn.execute(text('SELECT * FROM mangrovision.analyses ORDER BY analyzed_at, id')).mappings())
            points_before = list(conn.execute(text('SELECT id, analysis_id FROM mangrovision.planting_points ORDER BY id')))
            if not args.verify_applied:
                path = ROOT / 'alembic/versions/20260916_0007_analysis_numbering.py'
                spec = importlib.util.spec_from_file_location('numbering_migration', path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                module.op = Operations(MigrationContext.configure(conn))
                module.upgrade()
            after = list(conn.execute(text('SELECT * FROM mangrovision.analyses ORDER BY analyzed_at, id')).mappings())
            assert len(before) == len(after)
            for index, (old, new) in enumerate(zip(before, after), 1):
                if not args.verify_applied:
                    assert new['image_name'] == f'Analysis {index}'
                    assert new['analysis_number'] == index
                assert new['image_name'] == f"Analysis {new['analysis_number']}"
                for key in old:
                    if key not in {'image_name', 'analysis_number', 'analysis_detail_json'}:
                        assert old[key] == new[key], key
                old_detail = old['analysis_detail_json'] or {}
                new_detail = new['analysis_detail_json']
                for key in old_detail:
                    assert old_detail[key] == new_detail[key], key
                assert new_detail['source_image_name'] == old_detail.get('source_image_name', old['image_name'])
            points_after = list(conn.execute(text('SELECT id, analysis_id FROM mangrovision.planting_points ORDER BY id')))
            assert points_before == points_after
            sequence = conn.execute(text('SELECT last_value, is_called FROM mangrovision.analysis_number_seq')).mappings().one()
            next_number = sequence['last_value'] + int(sequence['is_called'])
            assert next_number > max((row['analysis_number'] for row in after), default=0)
            print({'mode': 'verified applied migration' if args.verify_applied else 'dry run (rolled back)',
                   'names': [row['image_name'] for row in after], 'next_name': f'Analysis {next_number}',
                   'unchanged_planting_point_links': len(points_after)})
        finally:
            transaction.rollback()
    engine.dispose()


if __name__ == '__main__':
    main()
