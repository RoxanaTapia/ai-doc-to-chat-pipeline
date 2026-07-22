# Time-limited pilot invites

HMAC-signed tokens for `ai-doc-pilot` `/app`, plus auto email on **Request an invite**.
Caddy basic auth remains the operator **Login** fallback.

## Setup

1. Repo-root `.env`:

```bash
INVITE_SECRET=$(openssl rand -hex 32)
INVITE_BASE_URL=https://ai-doc-pilot.roxanatapia.dev
INVITE_REQUEST_TTL=72h
INVITE_NOTIFY_TO=hello@roxanatapia.dev

SMTP_HOST=smtp.example.com
SMTP_PORT=587
SMTP_USER=hello@roxanatapia.dev
SMTP_PASSWORD=…
SMTP_FROM=hello@roxanatapia.dev
SMTP_TLS=true
```

2. Recreate (repo root):

```bash
docker compose --env-file .env -p ai-doc-to-chat-pipeline \
  -f deploy/docker-compose.yml -f deploy/docker-compose.caddy.yml up -d --build
```

## Request flow (visitors)

1. Gate → **Request an invite** → email
2. `POST /invite/request` mints a TTL token, emails the visitor (Jinja HTML), notifies `INVITE_NOTIFY_TO`
3. Rate limit: 3 requests / hour per IP and per email
4. Visitor opens the link or pastes the code under **I have an invite**

The browser response never includes the token.

## Manual mint

```bash
python deploy/invite/mint.py --ttl 72h --label client-acme
```

## Revoke

Rotate `INVITE_SECRET` and recreate the `invite` service.
