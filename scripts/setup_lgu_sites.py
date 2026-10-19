"""Initialize permanent LGU access after migration 20261019_0020.

Credentials are encrypted in the database and shown only in the staff workspace.
This command never prints the organization code or private configuration.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mangrovision_db.lgu_sites import ensure_lgu_account


def main():
    try:
        ensure_lgu_account()
    except Exception as error:
        print(f'LGU setup stopped ({type(error).__name__}); private details suppressed.', file=sys.stderr)
        return 1
    print('Permanent LGU organization and planter account are ready.')
    print('Staff can view the username and permanent organization code in Monitoring > LGU Sites.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
