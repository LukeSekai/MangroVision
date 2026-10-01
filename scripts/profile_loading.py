"""Read-only loading benchmark. Prints timings/counts, never row data or credentials.

Run with the project's Python. Optional --baseline-functions accepts an inspect
source snapshot for comparing old and new monitoring functions on the same data.
"""
import argparse
import gzip
import json
import statistics
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import event, text
import planting_database as db
from mangrovision_db.compat import get_engine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-functions', type=Path)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--dashboard', action='store_true')
    parser.add_argument('--site-id', type=int)
    parser.add_argument('--transport', action='store_true', help='Measure lossless JSON/gzip response sizes.')
    args = parser.parse_args()
    engine = get_engine()
    queries = []

    @event.listens_for(engine, 'begin')
    def readonly(conn):
        conn.exec_driver_sql('SET TRANSACTION READ ONLY')
        conn.exec_driver_sql("SET LOCAL statement_timeout = '15s'")

    @event.listens_for(engine, 'before_cursor_execute')
    def track(conn, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith(('SELECT', 'WITH')):
            queries.append((statement, parameters))

    # Warm the connection pool; report query work separately from cold TLS setup.
    with engine.connect() as conn:
        conn.execute(text('SELECT 1'))
    functions = [('monitoring', db.list_organization_monitoring_summaries),
                 ('history', db.list_organization_monitoring_records),
                 ('map_points', db.list_planter_assignment_map_points)]
    dashboard_time = db._manila_now()
    dashboard_filters = dict(date_from=f'{dashboard_time.year}-01-01',
                             date_to=dashboard_time.date().isoformat(), as_of=dashboard_time.isoformat())
    if args.site_id is not None:
        dashboard_filters['site_id'] = args.site_id
    if args.dashboard:
        functions = [(key, lambda getter=getattr(db, 'get_dashboard_' + key): getter(**dashboard_filters))
                     for key in ('overview', 'operations', 'ecology', 'sites')]
    baseline = None
    if args.baseline_functions:
        baseline = dict(vars(db))
        for source in json.loads(args.baseline_functions.read_text(encoding='utf8')).values():
            exec(compile(source, '<baseline>', 'exec'), baseline)
        if args.dashboard:
            functions[0:0] = [('baseline_' + key, lambda getter=baseline['get_dashboard_' + key]: getter(**dashboard_filters))
                              for key in ('overview', 'operations', 'ecology', 'sites')]
        else:
            functions[0:0] = [('baseline_monitoring', baseline['list_organization_monitoring_summaries']),
                              ('baseline_history', baseline['list_organization_monitoring_records'])]
            if 'list_planter_assignment_map_points' in json.loads(args.baseline_functions.read_text(encoding='utf8')):
                functions.insert(2, ('baseline_map_points', baseline['list_planter_assignment_map_points']))
    results, outputs = {}, {}
    for name, function in functions:
        timings, counts, captured = [], [], []
        for _ in range(max(1, args.runs)):
            queries.clear()
            started = perf_counter()
            output = function()
            timings.append(perf_counter() - started)
            counts.append(len(queries))
            captured = list(queries)
        outputs[name] = output
        plans = []
        # Inspect each distinct read once. EXPLAIN executes only SELECTs within
        # read-only transactions and does not change table/index definitions.
        with engine.connect() as conn:
            for statement, parameters in dict((sql, params) for sql, params in captured).items():
                plan = conn.exec_driver_sql('EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) ' + statement, parameters).scalar()[0]
                plans.append(round(plan['Execution Time'], 3))
        results[name] = {'median_seconds': round(statistics.median(timings), 3),
                         'select_queries': counts, 'rows': len(output),
                         'server_execution_ms': plans}
        if args.transport:
            from fastapi.encoders import jsonable_encoder
            from starlette.responses import JSONResponse
            body = JSONResponse(jsonable_encoder(output)).body
            compressed = gzip.compress(body, compresslevel=3)
            assert gzip.decompress(compressed) == body
            results[name].update(json_bytes=len(body), gzip_bytes=len(compressed))
        print(json.dumps({name: results[name]}), flush=True)
    if baseline:
        for key in outputs:
            if 'baseline_' + key not in outputs:
                continue
            if outputs[key] != outputs['baseline_' + key]:
                raise AssertionError(key + ' differs from baseline; no row data printed')
        print('All compared loading results exactly match baseline.')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('Profiling failed: ' + type(error).__name__, file=sys.stderr)
        sys.exit(1)
