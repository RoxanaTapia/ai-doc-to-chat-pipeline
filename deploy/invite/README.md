# Invite service — multi-site

HMAC-signed, time-limited invite tokens for `/app` on both the AI Doc pilot and Receipt Intelligence hosts.
One service handles both sites; site is resolved per-request from the `Host` header.
Caddy basic auth remains the operator **Login** fallback.

## Supported sites

| Key | Cookie | Default host | Product |
|-----|--------|--------------|---------|
| `pilot` | `pilot_invite` | `ai-doc-pilot.roxanatapia.dev` | AI Doc pilot |
| `receipt` | `receipt_invite` | `receipt-intelligence.roxanatapia.dev` | Receipt Intelligence |

## Site resolution order

1. **Host header** — matched against the configured `base_url` hostname for each site.
2. **`site` POST field** — `pilot` or `receipt` sent by the gate form (fallback when Host is ambiguous, e.g. direct IP access).
3. **Default: `pilot`** — backward-compatible; existing codes keep working.

Old tokens (minted before multi-site) have no `site` claim and are treated as `pilot`.

## Env contract

Add to repo-root `.env` (`.env.example` is owned by config-guardian):

```bash
# Shared
INVITE_SECRET=$(openssl rand -hex 32)
INVITE_REQUEST_TTL=72h

# AI Doc pilot (existing)
INVITE_BASE_URL=https://ai-doc-pilot.roxanatapia.dev
INVITE_NOTIFY_TO=hello@roxanatapia.dev   # shared notify unless overridden

# Receipt Intelligence (new)
RECEIPT_INVITE_BASE_URL=https://receipt-intelligence.roxanatapia.dev
# RECEIPT_INVITE_NOTIFY_TO=team@example.com  # optional; falls back to INVITE_NOTIFY_TO

# SMTP (shared for both sites)
SMTP_HOST=smtp.example.com
SMTP_PORT=587
SMTP_USER=hello@roxanatapia.dev
SMTP_PASSWORD=…
SMTP_FROM=hello@roxanatapia.dev
SMTP_TLS=true
```

## Start

```bash
docker compose --env-file .env -p ai-doc-to-chat-pipeline \
  -f deploy/docker-compose.yml -f deploy/docker-compose.caddy.yml up -d --build
```

## Request flow (visitors)

1. Gate → **Request an invite** → email
2. `POST /invite/request` resolves site from Host, mints a TTL token embedding the site claim, emails the visitor (branded per site), notifies the site's `notify_to` address
3. Rate limit: 3 requests / hour per IP and per email (shared across sites)
4. Visitor opens the link or pastes the code under **I have an invite**

The browser response never includes the token.

## Manual mint

```bash
# AI Doc pilot (default)
python deploy/invite/mint.py --ttl 72h --label client-acme

# Receipt Intelligence
python deploy/invite/mint.py --site receipt --ttl 24h --label client-beta

# Override URL explicitly
python deploy/invite/mint.py --site receipt --base-url https://receipt-intelligence.roxanatapia.dev --ttl 48h
```

## Smoke-test (local / no SMTP)

Start the server with a dummy secret:

```bash
cd deploy/invite
INVITE_SECRET=localtestonly INVITE_BASE_URL=http://localhost:8090 \
  RECEIPT_INVITE_BASE_URL=http://localhost:8090 python server.py &
```

Mint a receipt token:

```bash
INVITE_SECRET=localtestonly python mint.py --site receipt --ttl 72h
# → code=v1.<body>.<sig>   url=http://localhost:8090/invite/redeem?token=…
```

Redeem via Host-spoofed curl (simulates receipt host):

```bash
TOKEN=<paste code here>
curl -si -H "Host: receipt-intelligence.roxanatapia.dev" \
  "http://localhost:8090/invite/redeem?token=$TOKEN"
# Expect: 303 Location=/app  Set-Cookie: receipt_invite=…
```

Verify the receipt cookie:

```bash
curl -si -H "Host: receipt-intelligence.roxanatapia.dev" \
  -H "Cookie: receipt_invite=$TOKEN" \
  http://localhost:8090/verify
# Expect: 200 X-Invite-Ok: 1
```

Pilot path unchanged (no Host header → defaults to pilot):

```bash
INVITE_SECRET=localtestonly python mint.py --site pilot --ttl 72h
TOKEN=<paste code>
curl -si "http://localhost:8090/invite/redeem?token=$TOKEN"
# Expect: 303  Set-Cookie: pilot_invite=…
```

## Revoke

Rotate `INVITE_SECRET` and recreate the `invite` service. All outstanding tokens become invalid immediately.
