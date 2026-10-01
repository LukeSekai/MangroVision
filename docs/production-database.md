# Production database operations

## Environment model

Use Supabase-managed PostgreSQL/PostGIS and a private Supabase Storage bucket
for the online capstone. Use the pinned Docker Compose services for development
and tests. The Supabase free project is a capstone environment, not an
operational LGU production tier: move real records to a paid managed service
with automatic backups, tested restores, uptime ownership, monitoring, and a
named support operator.

FastAPI is the only database and storage client. Never place PostgreSQL,
Supabase service-role, or S3 credentials in the React build.

## Local PostgreSQL/PostGIS and MinIO

Copy `.env.example` to `.env`, generate a long MinIO secret, and keep the same
values in `.env` and `docker-compose.yml`. Then run:

```powershell
docker compose up -d
alembic upgrade head
python scripts/sync_gis_coverage.py
python scripts/bootstrap_admin.py
```

The PostgreSQL image is pinned to major version 17 with PostGIS 3.5. Match the
PostgreSQL major version to the selected Supabase project before deployment.

## Supabase provisioning

1. Create a project in the nearest available Southeast Asian region.
2. Enable PostGIS and create a private `analysis-images` bucket.
3. Create a restricted application login and a separate owner/migration login.
4. Give `DATABASE_URL` the session-pooler URL used by FastAPI.
5. Give `MIGRATION_DATABASE_URL` the direct owner URL used only by Alembic,
   migration, backup, and restore commands.
6. Set `DB_SSLMODE=require`, `COOKIE_SECURE=true`, and exact HTTPS origins in
   `TRUSTED_ORIGINS`.
7. Configure the Supabase S3-compatible endpoint and server-side credentials.
8. Run `alembic upgrade head`; readiness must report the repository's current migration head.

For the hosted project, copy `.env.example` to the Git-ignored `.env` and set:

- `MIGRATION_DATABASE_URL` to the owner/direct connection shown by **Connect**.
  If the local network has no IPv6, use the session pooler on port 5432 for the
  capstone cutover.
- `DATABASE_URL` to the session-pooler URL on port 5432, replacing its username
  with `mangrovision.<project-ref>` and using a new password chosen only for the
  restricted application role.
- `S3_ENDPOINT_URL` to the project's Storage S3 endpoint, plus the server-only
  access-key ID, secret, region, and private bucket name from Storage settings.

Do not paste these values into chat, put them in a React `.env` file, or pass
them as command-line arguments. Once `.env` is complete, provision and verify
the remaining server-only resources without printing credentials:

```powershell
python scripts/configure_app_role.py
python scripts/bootstrap_storage.py --create
```

Rotate any secret that has ever appeared in tracked configuration. Do not
reuse local example passwords.

## Migration dry run

Stop writers or work from a frozen copy. The command itself takes a consistent
read-only SQLite snapshot before inspecting data:

```powershell
python scripts/migrate_sqlite_to_postgres.py `
  --source .\planting_zones.db `
  --dry-run `
  --manifest .\migration-manifests\source-dry-run.json
```

Naive SQLite timestamps are interpreted as `Asia/Manila` by default and stored
as UTC `TIMESTAMPTZ`. Override with `--naive-timezone UTC` only if the source
installation is known to have written naive UTC values.

## Final migration and verification

First make an encrypted backup. Then stop API writes, preserve SQLite and the
active GeoJSON files as a read-only rollback archive, and run:

```powershell
alembic upgrade head
python scripts/migrate_sqlite_to_postgres.py `
  --source .\planting_zones.db `
  --eroded .\eroded_zones.geojson `
  --forbidden .\new_forbidden.geojson `
  --manifest .\migration-manifests\cutover.json
python scripts/migrate_sqlite_to_postgres.py `
  --source .\planting_zones.db `
  --verify-only `
  --manifest .\migration-manifests\verify.json
```

The import is idempotent by preserved primary key and source fingerprint. It
can resume a failed run, skips all old `auth_sessions`, resets identity
sequences, validates SQLite foreign keys and PostGIS geometry, and checks every
private object against its byte size and SHA-256 metadata.

Before reopening access, smoke-test login, processing/preflight, saving,
signed image display, map layers, planning, aggregate monitoring, schedules,
dashboard, exports, and deletions. If verification fails before reopening,
restore the archived SQLite deployment. After PostgreSQL accepts new writes,
use Alembic fix-forward migrations rather than writing to both databases.

## Encrypted backup and restore

Install PostgreSQL client tools and OpenSSL. Set a strong secret in the process
environment, not a command argument:

```powershell
$env:BACKUP_ENCRYPTION_PASSWORD = '<long secret from password manager>'
python scripts/backup_production.py --output-dir D:\MangroVisionBackups
```

Backups contain a custom-format `pg_dump`, every private bucket object, and a
SHA-256 manifest, encrypted with AES-256-CBC/PBKDF2. Keep a copy outside
Supabase. For the capstone, back up before every migration/demo and weekly
during active data entry.

Restore into a prepared PostgreSQL database and private bucket only after
checking the target environment:

```powershell
$env:BACKUP_ENCRYPTION_PASSWORD = '<same secret>'
python scripts/restore_production.py `
  D:\MangroVisionBackups\mangrovision-YYYYMMDDTHHMMSSZ.tar.enc `
  --confirm-restore
```

The restore command uses `pg_restore --clean --if-exists`; the explicit flag is
required because this replaces database objects represented by the archive.
It verifies all archived checksums before changing the target.

## Routine maintenance

Monitoring rounds remain anchored to 14 calendar days from each point's actual
planting date (Asia/Manila). A round falling on Saturday or Sunday is carried
forward to Monday; later rounds still use planting-day multiples of 14. Staff
receive in-app reminders one day before, on the scheduled weekday, and while
monitoring is overdue. The app refreshes those notices when an LGU user opens
it. Email reminders go to the LGU inbox three days before, one day before,
and on the scheduled weekday, but phases falling on a weekend are skipped.
For this Supabase project, use
`docs/supabase-cron-reminders.md` to send email without anyone opening the app.
As an alternative on another always-on cloud worker, run the Python/SMTP job
once per weekday in the Asia/Manila morning (for example, 08:00):

```powershell
python scripts/send_monitoring_reminders.py
```

Set `LGU_REMINDER_EMAIL=mangrovision.lgu@gmail.com` and configure
`SMTP_HOST`, `SMTP_PORT`, `SMTP_FROM`, `SMTP_USERNAME`, and `SMTP_PASSWORD` in
the backend's private environment. For Brevo, use
`SMTP_HOST=smtp-relay.brevo.com`, `SMTP_PORT=587`, and the SMTP login/key from
Brevo's SMTP settings; `SMTP_FROM` must be a verified sender. The job exits
nonzero if SMTP is unconfigured or sending fails. Do not put SMTP credentials
in the React build or commit them to Git. The Python job needs access to the
same PostgreSQL database as the app and an always-on cloud host. Supabase Cron
instead calls an Edge Function with a separate Brevo **API key**. Do not
schedule both senders. See `docs/monitoring-email-setup.md` for SMTP testing.

Run this from a scheduler to retry committed deletion jobs and remove abandoned
processing previews that were never attached to an analysis:

```powershell
python scripts/cleanup_storage.py --preview-age-hours 24 --job-limit 100
```

Monitor `/api/health/live` for process liveness and `/api/health/ready` for
database connectivity, exact Alembic revision, and bucket accessibility.
