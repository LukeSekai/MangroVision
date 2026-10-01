# MangroVision setup on a new laptop

For a request to set up or run MangroVision on a groupmate's laptop, follow
[`docs/groupmate-setup.md`](docs/groupmate-setup.md) in order. The current app is
FastAPI plus React; the older Streamlit launcher is historical.

Check the two separately shared runtime assets, install the listed dependencies,
configure the private `.env` from `.env.example`, run the setup checker, and
start the app. Report any missing asset, credential, or dependency clearly.
Automated test suites are not required for setup.

Never commit or print private `.env` values. The database and private storage
must be configured for the groupmate to see the existing planting records.
