# SHAP-GSD Dashboard

Interactive graphical abstract for the SHAP-GSD paper.
Cycles through all 9 canonical case studies (one per attack class) with
per-case feature-group attribution, temporal neighbourhood, and node-novelty panels.

## Requirements

- Node.js ≥ 18
- npm ≥ 9

## Install

```bash
cd dashboard
npm install
```

## Run (development)

```bash
npm run dev
```

Opens at `http://localhost:5173`.

## Build (static export)

```bash
npm run build
```

Output goes to `dashboard/dist/`. Serve with any static file server, e.g.:

```bash
npx serve dist
```

## Run in Docker

```bash
cd dashboard
docker compose up --build
```

Opens at `http://localhost:8080`. The container is always named `shap-gsd-dashboard` (set in `docker-compose.yml`).

```bash
docker compose down   # stop and remove the container
docker compose up     # restart without rebuilding
```

The container serves the pre-built static bundle via nginx — no Node.js needed at runtime.

## Dependencies

| Package | Version | Role |
|---------|---------|------|
| react | ^19 | UI framework |
| react-dom | ^19 | DOM renderer |
| vite | ^8 | Build tool / dev server |
| @vitejs/plugin-react | ^6 | JSX + Fast Refresh |
| eslint | ^10 | Linter |
