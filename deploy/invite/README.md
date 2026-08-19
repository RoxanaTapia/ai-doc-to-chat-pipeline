# Invite service — pilot-only

HMAC-signed, time-limited invite tokens for `/app` on the AI Doc pilot host.
Site is resolved per-request from the `Host` header (defaulting to pilot).
Caddy basic auth remains the operator **Login** fallback.

On the portfolio VPS after cutover, mint from [roxanatapia-edge](https://github.com/RoxanaTapia/roxanatapia-edge). This tree stays for a dedicated VM and for rollback. Env keys stay in this repo's `.env`.

## Supported sites

| Key | Cookie | Default host | Product |
|-----|--------|--------------|---------|
| `pilot` | `pilot_invite` | `ai-doc-pilot.roxanatapia.dev` | AI Doc pilot |

## Site resolution order

1. **Host header** — matched against the configured `base_url` hostname.
2. **`site` POST field** — `pilot` sent by the gate form (fallback when Host is ambiguous, e.g. direct IP access).
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

# Operator notify timing: redeem (default) | request (legacy) | none
INVITE_NOTIFY_ON=redeem

# Cloudflare Turnstile (server-side verify). Site key is public and lives on
# gate HTML in roxanatapia-web — do not invent it here.
TURNSTILE_SECRET_KEY=
# Defaults to true when TURNSTILE_SECRET_KEY is set; set false for local smoke
# TURNSTILE_ENABLED=false

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
# Dedicated VM (this overlay):
docker compose --env-file .env -p ai-doc-to-chat-pipeline \
  -f deploy/docker-compose.yml -f deploy/docker-compose.caddy.yml up -d --build

# Portfolio VPS after cutover: invite runs in roxanatapia-edge.
# https://github.com/RoxanaTapia/roxanatapia-edge/blob/main/CUTOVER.md
```

## Request flow (visitors)

1. Gate → **Request an invite** → email (Turnstile widget + honeypot on the form; sister repo)
2. `POST /invite/request` resolves site from Host, checks honeypot / disposable domain / Turnstile, then mints a TTL token and emails the visitor (branded per site)
3. Rate limit: 3 requests / hour per IP and per email (shared across sites)
4. Visitor opens the link or pastes the code under **I have an invite**
5. On successful redeem, operator notify email is sent (default `INVITE_NOTIFY_ON=redeem`)

The browser response never includes the token.

### Spam defenses (#124)

| Check | Behavior |
|-------|----------|
| Cloudflare Turnstile | Required when enabled; missing/invalid → calm **400**, no mint/email/notify |
| Honeypot (`company` / `website`) | Filled → **200** success-looking response, no mint/email/notify |
| Disposable domains | Known throwaways → calm **400**, no mint/email/notify |
| Operator notify | Default **redeem-only**; `request` = legacy per-request; `none` = off |
| Structured logs | One JSON line per request/redeem attempt on stderr (`event`, `email`/`label`, `client_ip`, `user_agent`, `referer`, `site`, `outcome`, `http_status`) |

## Manual mint

```bash
# AI Doc pilot (default)
python deploy/invite/mint.py --ttl 72h --label client-acme
```

## Smoke-test (local / no SMTP)

Start the server with a dummy secret and Turnstile off (no Cloudflare widget locally):

```bash
cd deploy/invite
INVITE_SECRET=localtestonly INVITE_BASE_URL=http://localhost:8090 \
  TURNSTILE_ENABLED=false INVITE_NOTIFY_ON=none \
  python server.py &
```

With SMTP unset, `POST /invite/request` returns **503** (`smtp_unconfigured`) after spam checks — that is expected. Redeem / verify still work with minted tokens.

Pilot path unchanged (no Host header → defaults to pilot):

```bash
INVITE_SECRET=localtestonly python mint.py --site pilot --ttl 72h
TOKEN=<paste code>
curl -si "http://localhost:8090/invite/redeem?token=$TOKEN"
# Expect: 303  Set-Cookie: pilot_invite=…
```

### Exit demo

Visitors and operators leave the invited session via **`GET` or `POST` `/invite/exit`**. The service clears the `pilot_invite` cookie and responds **303** to `/` (public gate / home). Caddy already treats `/invite*` as public, so no proxy change is needed.

UI copy should say **Exit demo** or **Back to home**, not "Log out." Exit clears only the invite cookie; it does **not** clear browser Basic Auth credentials used by the operator **Login** fallback.

If `/app` is hit with a **stale or invalid** invite cookie, Caddy still matches the cookie header and calls `forward_auth` → `/verify`. Verify now clears that cookie and **303**s to `/` (same as Exit) instead of returning a bare `invalid invite` page that blocked Login.

```bash
# After a redeem smoke (cookie set), exit clears it and redirects home (pilot-only)
curl -si -H "Cookie: pilot_invite=$TOKEN" \
  http://localhost:8090/invite/exit
# Expect: 303 Location=/  Set-Cookie: pilot_invite=… (expired)
```

## Revoke

Rotate `INVITE_SECRET` and recreate the `invite` service. All outstanding tokens become invalid immediately.
