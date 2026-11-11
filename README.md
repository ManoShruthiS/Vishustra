# VISHUSTRA

**Where Attention Becomes Visible.**

VISHUSTRA is an interactive Transformer research laboratory. Its Python core
implements the architecture and calculations from *Attention Is All You Need*,
then exposes the intermediate values required to inspect and visualize them.

## What works today

- A tested, from-scratch Transformer implementation with encoder, decoder,
  multi-head attention, positional encoding, training utilities, experiments,
  and visualization modules.
- A FastAPI service that exposes paper-reference configurations and the full
  scaled dot-product attention calculation for a browser client.
- An experiment workspace API for creating, listing, and re-running
  reproducible experiment records.
- Deterministic attention analysis: using the same tokens, dimension, and seed
  always produces the same matrices.

## Run locally

Use Python 3.10 or later. From the repository root:

```powershell
python -m pip install -e ".[dev]"
python -m uvicorn app.main:app --app-dir backend --reload
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to explore the
interactive API documentation.

## Run the dashboard

With the API server running, open a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

The Vite development server proxies API requests to `http://127.0.0.1:8000`.
For a separately hosted API, set `VITE_API_BASE_URL` before building the
dashboard.

## API examples

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/api/configs
Invoke-RestMethod http://127.0.0.1:8000/api/experiments
```

```powershell
@{
  tokens = @("the", "scientist", "studied")
  d_k = 4
  seed = 11
} | ConvertTo-Json | Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/api/attention/analyze `
  -ContentType "application/json"
```

The attention endpoint returns the raw scores, scaled scores, softmax weights,
and final output so a future web interface can show each stage of Equation 1.

Use `POST /api/experiments` to register an experiment. The request accepts a
name, hypothesis, seed, and partial `dataset`, `architecture`, or `training`
overrides. `POST /api/experiments/{id}/rerun` creates a fresh experiment record
with the same configuration, preserving the project’s reproducibility trail.
`POST /api/experiments/{id}/run` then executes the registered experiment and
returns the completed record with its collected metrics.

## Verify

```powershell
python -m pytest
python -m ruff check backend
```
