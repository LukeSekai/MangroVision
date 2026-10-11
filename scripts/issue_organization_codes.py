"""Preview existing participants or issue codes after a verified backup and migration.

Never prints codes and never sends mail. Private handovers stay in ignored folders.
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
from mangrovision_db.planter_codes import ensure_codes, _delivery_guard, refresh_pending_code_emails
from planting_database import _get_connection, append_activity


def existing_accounts(conn):
    """Existing registered owners only, never historical merged or reserved owners."""
    return [dict(row) for row in conn.execute('''SELECT p.id AS planter_id, p.organization_id,
        o.name AS organization, p.participant_count,
        (SELECT count(*) FROM organization_participants op WHERE op.planter_id=p.id
            AND op.slot <= p.participant_count) AS slots
        FROM planters p JOIN organizations o ON o.id=p.organization_id
        WHERE p.status='active' AND p.merged_into_planter_id IS NULL
          AND p.username IS NOT NULL AND p.password_hash IS NOT NULL ORDER BY p.id''').fetchall()]


def private_path(value):
    path = Path(value).resolve()
    if not any(path.is_relative_to(ROOT / folder) for folder in ('tmp', 'backups')):
        raise ValueError('Save code handovers only in the ignored tmp or backups directories.')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def issue_existing(conn, *, expected_accounts, expected_slots):
    rows = existing_accounts(conn)
    if len(rows) != expected_accounts or sum(int(row['slots']) for row in rows) != expected_slots:
        raise ValueError('Organizations or participants changed. Review a fresh preview before issuing codes.')
    for row in rows:
        if int(row['slots']) != int(row['participant_count']):
            raise ValueError('An organization has missing participant slots. Resolve that before issuing codes.')
    result = []
    for row in rows:
        _delivery_guard(conn, row['organization_id'])
        codes = ensure_codes(conn, row['planter_id'])
        refresh_pending_code_emails(conn, row['organization_id'], codes)
        append_activity(conn, action='participant.codes_issued', actor_type='system',
            organization_id=row['organization_id'], summary='Issued individual codes for existing organization participants.',
            details={'planter_id': row['planter_id'], 'participant_count': row['participant_count']})
        result.append({**row, 'codes': codes})
    return result


def handover(accounts):
    e = lambda value: html.escape(str(value), quote=True)
    sections = []
    for account in accounts:
        rows = ''.join(f"<tr><td>Participant {int(entry['participant_slot'])}</td><td><code>{e(entry['code'] or 'Revoked')}</code></td></tr>"
                       for entry in account['codes'])
        sections.append(f"<section><h2>{e(account['organization'])}</h2><table><tr><th>Participant</th><th>Private code</th></tr>{rows}</table></section>")
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Private planter codes</title>'
        '<style>body{font:16px Arial;margin:32px;color:#153f30}table{border-collapse:collapse;width:100%;max-width:620px}'
        'th,td{border:1px solid #ccdcd4;padding:10px;text-align:left}th{background:#edf7f1}code{font-size:18px}'
        'section{break-inside:avoid;margin:28px 0}</style><h1>Private planter codes</h1>'
        '<p>Keep each organization’s full list with its leader. Give each participant only their own code, '
        'matching their existing participant number. Codes work across activities and browsers, '
        'restoring the same participant number and assigned points.</p>'
        + ''.join(sections) + '</html>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--issue', action='store_true')
    parser.add_argument('--expected-accounts', type=int)
    parser.add_argument('--expected-slots', type=int)
    parser.add_argument('--verified-backup')
    parser.add_argument('--encrypted-handover')
    parser.add_argument('--private-html')
    args = parser.parse_args()
    conn = None
    try:
        conn = _get_connection()
        if not args.issue:
            rows = existing_accounts(conn)
            print(json.dumps({'organizations': len(rows), 'participant_slots': sum(int(row['slots']) for row in rows),
                'slot_counts_match': all(int(row['slots']) == int(row['participant_count']) for row in rows),
                'database_changed': False}))
            return
        if not all((args.verified_backup, args.encrypted_handover, args.private_html,
                    args.expected_accounts, args.expected_slots)):
            raise ValueError('Issuance requires reviewed counts, a verified encrypted backup and private handover paths.')
        archive = private_path(args.verified_backup)
        if archive.suffix != '.enc' or not archive.is_file() or archive.stat().st_size == 0:
            raise ValueError('The verified encrypted backup is missing.')
        encrypted_path, html_path = private_path(args.encrypted_handover), private_path(args.private_html)
        if encrypted_path.exists() or html_path.exists():
            raise ValueError('Choose new handover paths; existing private files must not be replaced.')
        result = issue_existing(conn, expected_accounts=args.expected_accounts, expected_slots=args.expected_slots)
        # Save encrypted material before commit. A failed commit leaves an artifact
        # requiring comparison to the current staff code list before distribution.
        encrypted = _cipher().encrypt(json.dumps(result).encode('utf-8'))
        with encrypted_path.open('xb') as output:
            output.write(encrypted)
        conn.commit()
        with html_path.open('x', encoding='utf-8') as output:
            output.write(handover(result))
        print(f'Issued codes for {len(result)} organizations. Private handover saved; no emails sent.')
    except Exception as error:
        if conn:
            conn.rollback()
        print(f'Code issuance did not finish ({type(error).__name__}); private details suppressed.', file=sys.stderr)
        return 1
    finally:
        if conn:
            conn.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
