#!/usr/bin/env sh
# Generate a self-signed TLS cert/key for a VPS public IP (Caddyfile.ip).
# Usage: ./deploy/generate-ip-tls.sh 1.2.3.4

set -eu

ip="${1:?public IPv4 required}"
out_dir="deploy/certs"

mkdir -p "$out_dir"

openssl req -x509 -newkey rsa:2048 \
  -keyout "${out_dir}/key.pem" \
  -out "${out_dir}/cert.pem" \
  -days 365 -nodes \
  -subj "/CN=${ip}" \
  -addext "subjectAltName=IP:${ip}"

# Caddy reads key as non-root — must be world-readable.
chmod 644 "${out_dir}/key.pem" "${out_dir}/cert.pem"

echo "Wrote ${out_dir}/cert.pem and ${out_dir}/key.pem for IP ${ip}."
