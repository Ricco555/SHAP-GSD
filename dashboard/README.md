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

# build the image
docker build -t shap-gsd-dashboard .

# run on port 8080 (--name keeps Docker from generating a random container name)
docker run --name shap-gsd-dashboard -p 8080:80 shap-gsd-dashboard
```

Opens at `http://localhost:8080`. The container serves the pre-built static bundle via nginx — no Node.js needed at runtime.

## Dependencies

| Package | Version | Role |
|---------|---------|------|
| react | ^19 | UI framework |
| react-dom | ^19 | DOM renderer |
| vite | ^8 | Build tool / dev server |
| @vitejs/plugin-react | ^6 | JSX + Fast Refresh |
| eslint | ^10 | Linter |
