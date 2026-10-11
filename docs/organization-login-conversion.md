# Existing organization login consistency

**Historical conversion:** Code-only planter sign-in was activated on October 12
and has superseded this username/password handover. Use
[Planter code access](planter-code-access.md) for the current conversion plan. Do
not distribute the old password handover after code-only activation.

The user approved the conversion, and all nine existing, active organization
accounts have been updated. New passwords are stored in the encrypted plan
under the Git-ignored `backups/` directory and the private local handover at
`tmp/organization-logins-private.html`.

| Organization | Current username |
| --- | --- |
| Leganes Mangrove Guardians Association | `leganes-mangrove-guardians-association` |
| Iloilo Coastal Youth Volunteers | `iloilo-coastal-youth-volunteers` |
| Katunggan Community Planters Cooperative | `katunggan-community-planters-cooperative` |
| INHS | `inhs` |
| NASUGBAN | `nasugban` |
| CICT | `cict` |
| Oton Youth | `oton-youth` |
| La Paz Youth | `la-paz-youth` |
| San Isidro Planters | `san-isidro-planters` |

Passwords use the same generator as new website accounts: an organization
prefix, four readable words, and eight independently generated digits.
For example, Oton Youth passwords begin `OtonYouth-`. Exact passwords are
excluded from previews, logs, and Git.

## What applying the conversion changes

- Each selected account's username and password hash change. Old passwords
  stop working, so leaders need the new login before their next sign-in.
- Account IDs, organization links, participant counts, device bindings,
  assignment ownership, planting statuses, monitoring history, and existing
  server sessions are preserved.
- Merged historical accounts and organizations without a login are skipped.
  Inactive accounts stay inactive if included in a future plan.
- A renamed login reuses a device identity already saved in that browser.
  The server matches keys only after authenticating the group. If more than
  one old identity matches, the user must choose using their recovery code.
  Different browsers or domains still require the existing recovery process.
- Unsent booking emails block conversion to prevent delivery of stale logins.
  No messages are sent by this operation. Previously delivered emails keep
  their original contents.
- All updates and credential-free audit records commit in one transaction.
  A conflict rolls back the batch. Reapplying the same plan does not reset
  passwords again.

## Activation procedure

1. Review the password-free preview in
   `tmp/organization-login-preview-20261012.json`.
2. Obtain approval for resetting existing credentials and distribute new
   details privately after activation. Approval for migration 0016 did not
   include existing account resets.
3. Publish the matching device compatibility frontend and restart the backend.
4. Pause writers; create and verify a fresh encrypted backup following
   [production database procedures](production-database.md).
5. Apply the reviewed encrypted plan. An account changed since preparation
   requires a fresh plan and review. No schema migration is needed.

```powershell
.\venv\Scripts\python.exe scripts/standardize_organization_logins.py `
  --apply --plan backups/organization-login-conversion-20261012.enc `
  --verified-backup backups/<fresh-verified-archive>.tar.enc
```

The `--verified-backup` argument requires the archive to exist; backup checksum
and readability verification are a separate required operator step. Keep
the verified archive and encrypted handover until the conversion is accepted.

To create a private, local handover after applying the plan:

```powershell
.\venv\Scripts\python.exe scripts/standardize_organization_logins.py `
  --plan backups/organization-login-conversion-20261012.enc `
  --export-private tmp/organization-logins-private.html
```

The tool verifies the credentials still match their accounts before exporting.
Share each row with its own organization leader. This HTML contains passwords;
keep it private and delete it after handover. The encrypted plan can be reopened
with the private `LIKE_EMAIL_ENCRYPTION_KEY` when needed.

## Verification

The temporary PostgreSQL tests modify session-local tables only and roll back
their changes. They verify credential conversion, unchanged allocations and
sessions, skipped accounts, username collisions, stale previews, idempotency,
unsent email conflicts, audit rollback, and device identity recovery.

```powershell
$env:MANGROVISION_RUN_TEMP_LIKE_TESTS = '1'
.\venv\Scripts\python.exe -m pytest tests/test_organization_credentials_postgres.py `
  tests/test_planter_activities.py tests/test_participant_devices_api.py -q
```

The client sign-in tests cover renamed usernames, failed login retries,
cookie recovery, and preserving other organizations' browser identities.
Frontend build and lint also pass. These regression checks changed no live
credentials and sent no real emails. The separately approved activation
updated the nine live accounts.

## Activation verification — October 12, 2026

The fresh encrypted backup is `backups/mangrovision-20261011T162822Z.tar.enc`.
Its database checksum and all 136 private-object checksums were verified;
`pg_restore --list` confirmed the database dump is readable. A full restore
test was not performed. Application and backup connections were verified
to target the same database before activation.

The nine accounts were updated in one transaction, with nine credential-free
audit entries. Current password hashes were verified against the private
handover. The before/after comparison covered all 32 existing database tables:
all original rows were preserved, with only the selected accounts' usernames
and password hashes changed. Existing participant devices, point ownership,
statuses, and sessions were unchanged. No schema migration was required.

The updated backend reports ready and includes renamed-device support.
The frontend sign-in suite passes 16 tests, including keeping an existing
session's device identity when its username changes. The matching frontend
was published to `https://mangrovision-user-testing.vercel.app` and checked
through the public site: the API reports ready; `/field` and `/landing-page`
return HTTP 200; the current field bundle includes renamed-device support;
anonymous activity requests still return HTTP 401.

Refresh the planter page to load the published sign-in update. Leaders need
their new login from the private handover; previously delivered confirmation
emails retain the old credentials. This conversion sent no replacement emails.
