# Remote user testing: Vercel frontend and laptop backend

The website is hosted on Vercel. Its `/api`, `/tiles`, and
`/monitoring_uploads` requests pass through Vercel to a Cloudflare Quick Tunnel
on the owner's laptop. The laptop runs the existing FastAPI, detector, and
orthophoto files, with the private database/storage settings in the root `.env`.
The expert needs the website URL and an authorized MangroVision account.

## Start the next testing session

From the repository root in PowerShell, using this laptop's existing venv:

```powershell
.\venv\Scripts\python.exe -X utf8 MangroVision_New/start_testing.py --deploy
```

The saved website URL is reused. On the first run on another laptop, provide it:

```powershell
.\venv\Scripts\python.exe -X utf8 MangroVision_New/start_testing.py --frontend-url https://mangrovision-user-testing.vercel.app --deploy
```

Wait for Vercel to finish deploying, then share the **website** URL, rather than
the temporary backend URL. Keep this terminal open, the laptop awake and
connected to the internet, and preferably plugged in. The launcher prevents
automatic Windows sleep while it runs; closing the lid or losing power can
still interrupt the session. It runs one API worker without development reload
and supervises the tunnel and API together. Starting either workspace launcher
again stops the previous managed stack before starting another one.

Each new Quick Tunnel has a different backend URL. `--deploy` updates the
Vercel routing and publishes a new frontend deployment behind the same website
URL. If deployment fails while the laptop API remains running, retry:

```powershell
.\venv\Scripts\python.exe -X utf8 MangroVision_New/deploy_testing.py
```

Stop with Ctrl+C, or from another terminal:

```powershell
.\venv\Scripts\python.exe -X utf8 MangroVision_New/start_testing.py --stop
```

Return to local development with `python MangroVision_New/start_dev.py`. Remote
testing ends when its API/tunnel stop; the Vercel frontend itself remains hosted.

## One-time setup on another laptop

1. Follow [groupmate-setup.md](groupmate-setup.md) and confirm backend readiness.
   Do not bootstrap accounts or migrate the already configured shared database.
2. Install Cloudflare's Windows `cloudflared` using its
   [official download instructions](https://developers.cloudflare.com/tunnel/downloads/).
   Put it on PATH, or save it as
   `MangroVision_New/.dev-server/tools/cloudflared.exe`. No domain is needed for
   [Quick Tunnels](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/).
3. Install and sign in to Vercel CLI. This keeps the CLI inside an ignored folder:

   ```powershell
   npm.cmd install --prefix tmp/vercel-cli --no-audit --no-fund vercel
   node tmp/vercel-cli/node_modules/vercel/dist/vc.js login
   node tmp/vercel-cli/node_modules/vercel/dist/vc.js link --yes --project mangrovision-user-testing --scope mangro-vision --cwd MangroVision_New/client
   ```

   Use the appropriate project/team if setting up a different deployment.
   Project settings: framework **Vite**, root directory **MangroVision_New/client**,
   Node **22.x**, output directory **dist**, build **npm run build**.
   Both local `.vercel` project metadata and credentials are excluded from Git.

## Routing and authentication

`client/vercel.mjs` reads the public `MANGROVISION_BACKEND_URL` at deployment
time. The root `vercel.mjs` exposes that same configuration to CLI deployments.
The root `.vercelignore` allows only frontend files to upload. Never upload
the laptop `.env`, GeoTIFF, checkpoint, or database credentials to Vercel.

The deployment script sets three public production variables:

| Variable | Value |
| --- | --- |
| `MANGROVISION_BACKEND_URL` | Current HTTPS Quick Tunnel origin |
| `VITE_API_BASE` | Empty; requests use the website's own origin |
| `VITE_TILE_SERVER` | `/tiles` |

This preserves the application's session and CSRF cookies on the frontend
origin. `start_testing.py` applies secure, host-only cookies and trusts the
exact configured frontend URL through process environment overrides. It does
not edit the private `.env`. Test the **production website URL**; a different
preview hostname is not automatically trusted by the backend.

Long image analysis uses `POST /api/analyses/jobs` followed by authenticated
`GET /api/analyses/jobs/{id}` progress polling. Quick Tunnels do not support SSE,
and long synchronous requests can exceed proxy timeouts. Polling retries
temporary connection failures without running AI again. If the initial upload
acknowledgement is lost, the client retries with the same request identifier;
the server returns the existing job instead of starting another analysis. HTML
gateway errors and interrupted JSON responses are handled as connection errors.
Only one background image analysis is accepted at a time on the laptop. Job
status/results are restricted to the user who started the job and expire after
one hour. Closing a tab does not cancel the laptop's running analysis. Unsaved
jobs/previews are held in memory and are lost on API restart; saved records
remain in the shared database. Saving remains an explicit user action.

## Before sending the URL

Open the website and sign in. Confirm that the map loads, an existing record is
visible, and one in-bounds geotagged image reaches its summary. Save or delete
records only when those changes are intended: this uses the existing shared DB.
The proxy's `/api/health/ready` must return `status: ready`; a health check alone
does not verify a complete signed-in workflow.

Logs and the current URLs are in the ignored `MangroVision_New/.dev-server`
folder: `testing-backend.log`, `testing-tunnel.log`, and `testing.json`.
If the website loads but its data fails, check the laptop terminal, these logs,
and whether the latest tunnel was deployed. Quick Tunnels are for temporary
testing and have no uptime guarantee. See
[Cloudflare's limits](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/)
and [Vercel external rewrites](https://vercel.com/docs/routing/rewrites).
