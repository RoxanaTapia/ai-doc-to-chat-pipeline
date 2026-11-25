#!/bin/sh
# Confirm this folder still matches the sums committed with VERSION.
# Demo CI runs this after copying the kit. Same idea as cmp in CI
# after scripts/sync-shared.sh: fail if a vendored file was edited.
set -eu
cd "$(dirname "$0")"
sha256sum --quiet -c SHA256SUMS
