#!/usr/bin/env sh
# Write deploy/caddy-basicauth.conf with a bcrypt hash for Caddy basic_auth.
# Usage: ./deploy/generate-caddy-auth.sh demo 'your-plain-password'

set -eu

user="${1:?username required}"
pass="${2:?plaintext password required}"
out="deploy/caddy-basicauth.conf"

hash="$(docker run --rm caddy:2.9.1-alpine caddy hash-password --plaintext "$pass")"

mkdir -p deploy
{
  printf 'basic_auth {\n'
  printf '\t%s %s\n' "$user" "$hash"
  printf '}\n'
} >"$out"

echo "Wrote $out for user '$user'."
