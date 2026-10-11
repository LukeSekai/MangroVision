"""Prepare/review existing logins, then apply the reviewed encrypted plan.

No credentials are printed. Real account changes require --apply and a fresh,
separately verified encrypted backup. This tool never sends emails.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mangrovision_db.appointment_email import _cipher
from mangrovision_db.organization_credentials import apply_conversion, prepare_conversion, public_preview
from planting_database import _get_connection
from mangrovision_db.passwords import verify_password


def private_path(value):
    path = Path(value).resolve()
    if not any(path.is_relative_to(ROOT / directory) for directory in ('backups', 'tmp')):
        raise ValueError('Save login artifacts only under the Git-ignored backups or tmp directories.')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def handover(plan):
    escape = lambda value: html.escape(str(value))
    rows = ''.join(f"<tr><td>{escape(entry['organization'])}</td><td>{escape(entry['username'])}</td>"
                   f"<td><code>{escape(entry['password'])}</code></td></tr>" for entry in plan['accounts'])
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Private organization logins</title>'
            '<style>body{font:16px Arial,sans-serif;margin:32px;color:#153f30}table{border-collapse:collapse;width:100%}'
            'th,td{border:1px solid #ccdcd4;padding:12px;text-align:left}th{background:#edf7f1}code{font-size:14px}</style>'
            '<h1>Private organization logins</h1><p>Share each row only with its organization leader. '
            'Existing planting records and participant allocations are preserved.</p>'
            '<table><thead><tr><th>Organization</th><th>Username</th><th>Password</th></tr></thead>'
            f'<tbody>{rows}</tbody></table></html>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--prepare', action='store_true')
    actions.add_argument('--apply', action='store_true')
    actions.add_argument('--export-private', metavar='HTML_PATH')
    parser.add_argument('--plan', required=True, help='Encrypted plan path under backups/')
    parser.add_argument('--preview', help='Password-free JSON preview path under tmp/')
    parser.add_argument('--verified-backup', help='Fresh encrypted archive verified before applying')
    args = parser.parse_args()
    conn = None
    try:
        plan_path = private_path(args.plan)
        cipher = _cipher()
        conn = _get_connection()
        if args.prepare:
            if not args.preview:
                raise ValueError('--prepare requires --preview.')
            plan = prepare_conversion(conn)
            preview_path = private_path(args.preview)
            # Exclusive creation prevents accidental replacement of a handover.
            with plan_path.open('xb') as output:
                output.write(cipher.encrypt(json.dumps(plan).encode('utf-8')))
            with preview_path.open('x', encoding='utf-8') as output:
                json.dump(public_preview(plan), output, indent=2)
            print(f"Prepared {len(plan['accounts'])} account changes; database unchanged. Password-free preview: {preview_path.relative_to(ROOT)}")
            return
        plan = json.loads(cipher.decrypt(plan_path.read_bytes()))
        if args.export_private:
            done = conn.execute('''SELECT count(*) AS count FROM activity_logs WHERE action = ?
                AND details ->> 'batch_id' = ?''',
                                ('organization_account.credentials_standardized', plan['batch_id'])).fetchone()
            if int(done['count']) != len(plan['accounts']):
                raise ValueError('Apply this reviewed plan before exporting its login handover.')
            for entry in plan['accounts']:
                current = conn.execute('SELECT username, password_hash FROM planters WHERE id = ?',
                                       (entry['planter_id'],)).fetchone()
                if not current or current['username'] != entry['username'] or not verify_password(current['password_hash'], entry['password'])[0]:
                    raise ValueError('An account login changed after conversion. Do not distribute this outdated handover.')
            export_path = private_path(args.export_private)
            with export_path.open('x', encoding='utf-8') as output:
                output.write(handover(plan))
            print(f'Private handover saved: {export_path.relative_to(ROOT)}')
            return
        if not args.verified_backup:
            raise ValueError('--apply requires --verified-backup after backup verification.')
        backup_path = private_path(args.verified_backup)
        if not backup_path.is_file() or backup_path.suffix != '.enc' or backup_path.stat().st_size == 0:
            raise ValueError('The verified encrypted backup archive is missing.')
        result = apply_conversion(conn, plan)
        conn.commit()
        print(f"Updated {result['updated']} organization logins. Already applied: {result['already_applied']}.")
    except Exception as issue:
        if conn is not None:
            conn.rollback()
        # Unexpected exceptions/SQL parameters can include passwords or private URLs.
        message = str(issue) if type(issue) is ValueError else type(issue).__name__
        print(f'Login conversion stopped: {message}', file=sys.stderr)
        raise SystemExit(1) from None
    finally:
        if conn is not None:
            conn.close()


if __name__ == '__main__':
    main()
