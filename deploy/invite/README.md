# Time-limited pilot invites

HMAC-signed tokens for `ai-doc-pilot` `/app`, validated by a tiny stdlib service and Caddy `forward_auth`. Caddy basic auth remains the operator fallback.

## Setup

1. Add a long random secret to repo-root `.env` (never commit it):

```bash
INVITE_SECRET=$(openssl rand -hex 32)
```

2. Recreate the stack (repo root):

```bash
docker compose --env-file .env -p ai-doc-to-chat-pipeline \
  -f deploy/docker-compose.yml -f deploy/docker-compose.caddy.yml up -d --build
```

## Mint

```bash
python deploy/invite/mint.py --ttl 72h --label client-acme
```

Share `code=` or `url=` out of band. Default TTL is 72h.

## Invitee flow

1. Open the public gate.
2. **Have invite** → paste code (POST `/invite/redeem`) or open the signed URL.
3. Cookie `pilot_invite` is set; continue to `/app`.
4. Operators can still use basic auth on `/app` without an invite cookie.

## Revoke

Rotate `INVITE_SECRET` and recreate the `invite` service (invalidates all outstanding tokens). Prefer short TTLs for one-off demos.
