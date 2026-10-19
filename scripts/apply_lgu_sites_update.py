"""Apply the additive LGU Sites schema update without printing credentials."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alembic import command
from alembic.config import Config
from mangrovision_db.lgu_sites import ensure_lgu_account


def main():
    try:
        config = Config(str(ROOT / 'alembic.ini'))
        config.set_main_option('script_location', str(ROOT / 'alembic'))
        command.upgrade(config, '20261019_0020')
        ensure_lgu_account()
    except Exception as error:
        print(f'LGU Sites update stopped ({type(error).__name__}); credentials and connection details suppressed.', file=sys.stderr)
        return 1
    print('LGU Sites database update and permanent planter account are ready.')
    print('Credentials are available to staff in Monitoring > LGU Sites.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
