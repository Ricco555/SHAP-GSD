# ── stage 1: build ────────────────────────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

# ── stage 2: serve ────────────────────────────────────────────────────────────
# caddy:alpine has a significantly smaller CVE surface than nginx:alpine
FROM caddy:alpine

COPY --from=builder /app/dist /usr/share/caddy
COPY Caddyfile /etc/caddy/Caddyfile

EXPOSE ${PORT:-80}
