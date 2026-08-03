"""Site registry: maps pilot / receipt to per-site config loaded from env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse

PILOT = "pilot"
RECEIPT = "receipt"

VALID_SITES = (PILOT, RECEIPT)


@dataclass(frozen=True)
class SiteConfig:
    key: str
    cookie: str
    base_url: str
    product_name: str
    host_label: str
    access_phrase: str
    notify_to: str


def load_registry() -> dict[str, SiteConfig]:
    """Build the site registry from environment variables."""
    pilot_base = os.environ.get("INVITE_BASE_URL", "https://ai-doc-pilot.roxanatapia.dev").rstrip(
        "/"
    )
    receipt_base = os.environ.get(
        "RECEIPT_INVITE_BASE_URL", "https://receipt-intelligence.roxanatapia.dev"
    ).rstrip("/")
    shared_notify = os.environ.get("INVITE_NOTIFY_TO", "hello@roxanatapia.dev").strip()
    receipt_notify = os.environ.get("RECEIPT_INVITE_NOTIFY_TO", "").strip() or shared_notify

    return {
        PILOT: SiteConfig(
            key=PILOT,
            cookie="pilot_invite",
            base_url=pilot_base,
            product_name="AI Doc",
            host_label="AI Doc pilot",
            access_phrase="Access to the AI Doc pilot is by invitation.",
            notify_to=shared_notify,
        ),
        RECEIPT: SiteConfig(
            key=RECEIPT,
            cookie="receipt_invite",
            base_url=receipt_base,
            product_name="Receipt Intelligence",
            host_label="Receipt Intelligence",
            access_phrase="Access to Receipt Intelligence is by invitation.",
            notify_to=receipt_notify,
        ),
    }


def _host_to_site(host: str, registry: dict[str, SiteConfig]) -> str | None:
    """Strip port from Host header and match to a configured base_url hostname."""
    bare = host.split(":")[0].lower().strip()
    if not bare:
        return None
    for key, cfg in registry.items():
        configured_host = urlparse(cfg.base_url).hostname or ""
        if bare == configured_host:
            return key
    return None


def resolve_site(
    host_header: str,
    site_param: str | None,
    registry: dict[str, SiteConfig],
) -> SiteConfig:
    """
    Resolve the active site config using priority:
    1. Host header matched against configured base_url hostnames.
    2. Explicit `site` field (pilot|receipt) from POST body.
    3. Default: pilot (backward compatible).
    """
    from_host = _host_to_site(host_header or "", registry)
    if from_host:
        return registry[from_host]
    if site_param and site_param in registry:
        return registry[site_param]
    return registry[PILOT]
