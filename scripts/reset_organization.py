"""Preview or reset one explicitly named organization, preserving map points.

The reset requires an organization with no project site. Apply saves a private
rollback snapshot before changing rows; credentials are never printed.
"""
import argparse
import gzip
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sqlalchemy import text
from mangrovision_db import get_engine


def scope_queries(organization_id):
    accounts = 'SELECT id FROM planters WHERE organization_id=:org'
    assignments = f'SELECT id FROM planter_assignments WHERE planter_id IN ({accounts})'
    events = f'SELECT id FROM planting_events WHERE planter_id IN ({accounts}) OR assignment_id IN ({assignments})'
    points = f'SELECT planting_point_id FROM planter_assignment_points WHERE assignment_id IN ({assignments}) UNION SELECT planting_point_id FROM planting_events WHERE id IN ({events})'
    deaths = f'SELECT id FROM point_death_records WHERE planter_id IN ({accounts}) OR assignment_id IN ({assignments}) OR planting_event_id IN ({events})'
    records = 'SELECT id FROM organization_monitoring_records WHERE organization_id=:org'
    return {
        'organizations': 'id=:org',
        'planters': f'id IN ({accounts})',
        'organization_participants': f'planter_id IN ({accounts})',
        'auth_sessions': f"subject_type='planter' AND subject_id IN ({accounts})",
        'planter_assignments': f'id IN ({assignments})',
        'planter_assignment_points': f'assignment_id IN ({assignments})',
        'planting_events': f'id IN ({events})',
        'planting_points': f'id IN ({points})',
        'point_death_records': f'id IN ({deaths})',
        'monitoring_observations': f'planting_event_id IN ({events}) OR death_record_id IN ({deaths})',
        'organization_monitoring_records': f'id IN ({records})',
        'monitoring_death_locations': f'record_id IN ({records}) OR planting_event_id IN ({events}) OR death_record_id IN ({deaths})',
        'replanting_requests': f'planting_event_id IN ({events}) OR replacement_event_id IN ({events}) OR assignment_id IN ({assignments})',
        'activity_logs': f'organization_id=:org OR actor_planter_id IN ({accounts}) OR planting_event_id IN ({events})',
        'planting_schedules': 'organization_id=:org OR (organization_id IS NULL AND lower(trim(organization))=:name)',
        'staff_notifications': "event_key LIKE :notice_prefix",
        'email_reminders': "status IN ('pending','failed') AND body ~ :email_line",
    }


def prepare_reset(conn, organization_id, expected_name, *, lock=False):
    organization = conn.execute(text('SELECT * FROM organizations WHERE id=:org' + (' FOR UPDATE' if lock else '')),
                                {'org': organization_id}).mappings().one_or_none()
    if not organization or organization['name'].strip().casefold() != expected_name.strip().casefold():
        raise ValueError('The organization id and expected name must both match.')
    params = {'org': organization_id, 'name': expected_name.strip().lower(),
              'notice_prefix': f'organization:{organization_id}:%',
              'email_line': rf'(?mi)^- {re.escape(expected_name.strip())}\s*$'}
    if conn.scalar(text('SELECT COUNT(*) FROM project_sites WHERE organization_id=:org'), params):
        raise ValueError('This reset is restricted to an organization with no project site.')
    queries = scope_queries(organization_id)
    if lock:
        for table in ('planters','planter_assignments','planter_assignment_points','planting_points','planting_events'):
            conn.execute(text(f'SELECT id FROM {table} WHERE {queries[table]} ORDER BY id FOR UPDATE'), params).all()
    snapshot = {table: [dict(row) for row in conn.execute(text(f'SELECT * FROM {table} WHERE {where} ORDER BY ' +
                     ('planter_id,slot' if table=='organization_participants' else 'id')), params).mappings()]
                for table, where in queries.items()}
    point_ids = [row['id'] for row in snapshot['planting_points']]
    event_ids = [row['id'] for row in snapshot['planting_events']]
    assignment_ids = [row['id'] for row in snapshot['planter_assignments']]
    record_ids = [row['id'] for row in snapshot['organization_monitoring_records']]
    # Refuse cross-organization data loss, including replacement chains and
    # monitoring baselines. Use saved ids, never a name/geometry broad delete.
    guards = (
        ('SELECT COUNT(*) FROM planter_assignment_points WHERE planting_point_id=ANY(:points) AND released_at IS NULL AND NOT (assignment_id=ANY(:assignments))', {'points':point_ids,'assignments':assignment_ids}),
        ('SELECT COUNT(*) FROM planting_events WHERE planting_point_id=ANY(:points) AND closed_at IS NULL AND NOT (id=ANY(:events))', {'points':point_ids,'events':event_ids}),
        ('SELECT COUNT(*) FROM planting_events WHERE replaces_event_id=ANY(:events) AND NOT (id=ANY(:events))', {'events':event_ids}),
        ('SELECT COUNT(*) FROM organization_monitoring_records WHERE baseline_record_id=ANY(:records) AND organization_id<>:org', {'records':record_ids,'org':organization_id}),
        ('SELECT COUNT(*) FROM monitoring_death_locations l JOIN organization_monitoring_records r ON r.id=l.record_id WHERE l.planting_event_id=ANY(:events) AND r.organization_id<>:org', {'events':event_ids,'org':organization_id}),
        ('SELECT COUNT(*) FROM planters WHERE merged_into_planter_id=ANY(:accounts) AND organization_id<>:org', {'accounts':[row['id'] for row in snapshot['planters']],'org':organization_id}),
    )
    for query, guard_params in guards:
        if conn.scalar(text(query), guard_params):
            raise ValueError('Other organization records depend on these points; reset stopped without changes.')
    return snapshot, params


def save_backup(snapshot, organization_id, directory):
    directory.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({'organization_id': organization_id, 'created_at':datetime.now(timezone.utc).isoformat(),
                          'tables':snapshot},default=str,indent=2).encode('utf-8')
    path = directory / f'organization-{organization_id}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.json.gz'
    with gzip.open(path,'xb') as output:
        output.write(payload)
    with gzip.open(path,'rb') as check:
        if check.read() != payload:
            raise ValueError('Backup verification failed; reset stopped.')
    return path, hashlib.sha256(payload).hexdigest()


def apply_reset(conn, snapshot, params):
    before = unaffected_digests(conn, snapshot)
    # Delete in dependency order. All affected ids come from the locked snapshot.
    for table in ('monitoring_death_locations','replanting_requests','monitoring_observations',
                  'point_death_records','activity_logs','staff_notifications','planting_schedules',
                  'auth_sessions'):
        ids = [row['id'] for row in snapshot[table]]
        if ids:
            conn.execute(text(f'DELETE FROM {table} WHERE id=ANY(:ids)'),{'ids':ids})
    account_ids = [row['id'] for row in snapshot['planters']]
    if account_ids:
        conn.execute(text('DELETE FROM organization_participants WHERE planter_id=ANY(:ids)'),{'ids':account_ids})
    event_ids = [row['id'] for row in snapshot['planting_events']]
    if event_ids:
        # Every replacement in this chain belongs to the same reset scope.
        conn.execute(text('UPDATE planting_events SET replaces_event_id=NULL WHERE id=ANY(:ids)'),{'ids':event_ids})
        conn.execute(text('DELETE FROM planting_events WHERE id=ANY(:ids)'),{'ids':event_ids})
    for table in ('planter_assignment_points','planter_assignments'):
        ids = [row['id'] for row in snapshot[table]]
        if ids:
            conn.execute(text(f'DELETE FROM {table} WHERE id=ANY(:ids)'),{'ids':ids})
    point_ids = [row['id'] for row in snapshot['planting_points']]
    if point_ids:
        conn.execute(text("""UPDATE planting_points SET status='planned',planted_at=NULL,planted_date=NULL,
            death_at=NULL,death_reason=NULL,death_reason_category=NULL,death_notes=NULL WHERE id=ANY(:ids)"""),{'ids':point_ids})
    records = [row['id'] for row in snapshot['organization_monitoring_records']]
    if records:
        conn.execute(text('UPDATE organization_monitoring_records SET baseline_record_id=NULL WHERE id=ANY(:ids)'),{'ids':records})
        conn.execute(text('DELETE FROM organization_monitoring_records WHERE id=ANY(:ids)'),{'ids':records})
    if account_ids:
        conn.execute(text('UPDATE planters SET merged_into_planter_id=NULL WHERE id=ANY(:ids)'),{'ids':account_ids})
        conn.execute(text('DELETE FROM planters WHERE id=ANY(:ids)'),{'ids':account_ids})
    conn.execute(text('DELETE FROM organizations WHERE id=:org'),params)
    # Daily emails aggregate multiple organizations. Remove only this group's
    # line, preserving other organizations' queued reminders.
    for row in snapshot['email_reminders']:
        conn.execute(text("UPDATE email_reminders SET body=regexp_replace(body,:email_line,'','g') WHERE id=:id"),{**params,'id':row['id']})
        conn.execute(text("DELETE FROM email_reminders WHERE id=:id AND body !~ '(?m)^- '"),{'id':row['id']})
    if point_ids:
        invalid = conn.scalar(text("""SELECT COUNT(*) FROM planting_points WHERE id=ANY(:ids)
            AND (status<>'planned' OR planted_at IS NOT NULL OR planted_date IS NOT NULL OR death_at IS NOT NULL)"""),{'ids':point_ids})
        if invalid or conn.scalar(text('SELECT COUNT(*) FROM planting_points WHERE id=ANY(:ids)'),{'ids':point_ids}) != len(point_ids):
            raise ValueError('Point reset verification failed.')
    if conn.scalar(text('SELECT COUNT(*) FROM organizations WHERE id=:org'),params):
        raise ValueError('Organization removal verification failed.')
    if unaffected_digests(conn, snapshot) != before:
        raise ValueError('Unrelated records changed; reset stopped without committing.')


def unaffected_digests(conn, snapshot):
    """Compare all rows outside the captured scope without exporting their data."""
    result = {}
    for table,rows in snapshot.items():
        if table=='organization_participants':
            key='planter_id'
            ids=[row['id'] for row in snapshot['planters']]
            ordering='t.planter_id,t.slot'
        else:
            key='id'
            ids=[row['id'] for row in rows]
            ordering='t.id'
        result[table]=conn.scalar(text(f"""SELECT md5(COALESCE(string_agg(to_jsonb(t)::text,'|' ORDER BY {ordering}),''))
            FROM {table} t WHERE NOT (t.{key}=ANY(:ids))"""),{'ids':ids})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--organization-id',type=int,required=True)
    parser.add_argument('--expected-name',required=True)
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--backup-dir',type=Path,default=ROOT/'backups'/'organization-resets')
    args=parser.parse_args()
    try:
        with get_engine().begin() as conn:
            conn.execute(text('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ'))
            conn.execute(text("SET LOCAL lock_timeout='5s'"))
            conn.execute(text("SET LOCAL statement_timeout='20s'"))
            if not args.apply:
                conn.execute(text('SET TRANSACTION READ ONLY'))
            snapshot,params=prepare_reset(conn,args.organization_id,args.expected_name,lock=args.apply)
            print(json.dumps({'organization_id':args.organization_id,'row_counts':{table:len(rows) for table,rows in snapshot.items()}}))
            if args.apply:
                path,digest=save_backup(snapshot,args.organization_id,args.backup_dir)
                apply_reset(conn,snapshot,params)
        if args.apply:
            print(json.dumps({'reset':'committed','points_returned_to_planned':len(snapshot['planting_points']),
                              'backup':str(path),'sha256':digest}))
        else:
            print('Preview only; no rows changed.')
    except Exception as error:
        # Database exceptions can contain bound account/session data. Only a
        # controlled validation error is safe to print; never dump DB errors.
        print(str(error) if isinstance(error,ValueError) else f'Reset failed: {type(error).__name__}; no changes committed.')
        return 1
    return 0


if __name__=='__main__':
    raise SystemExit(main())
