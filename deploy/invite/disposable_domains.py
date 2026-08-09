"""Small reject list for disposable / throwaway email domains.

Keep this set short and maintainable. Fail closed primarily on known-bad
domains; cheap local-part heuristics are secondary.
"""

from __future__ import annotations

import re

# ~20 common throwaways — expand carefully; prefer precision over coverage.
DISPOSABLE_DOMAINS: frozenset[str] = frozenset(
    {
        "mailinator.com",
        "guerrillamail.com",
        "guerrillamailblock.com",
        "sharklasers.com",
        "grr.la",
        "guerrillamail.info",
        "spam4.me",
        "tempmail.com",
        "temp-mail.org",
        "temp-mail.io",
        "10minutemail.com",
        "10minutemail.net",
        "yopmail.com",
        "yopmail.fr",
        "trashmail.com",
        "trashmail.me",
        "discard.email",
        "discardmail.com",
        "mailnesia.com",
        "getnada.com",
        "nada.email",
        "maildrop.cc",
        "throwaway.email",
        "fakeinbox.com",
        "emailondeck.com",
        "moakt.com",
        "tmpmail.org",
        "tmpmail.net",
    }
)

# Local part that is only digits and unusually long (bot gibberish).
_LONG_DIGIT_LOCAL = re.compile(r"^\d{12,}$")
# Domain must have a plausible TLD of 2+ letters (rejects "user@host" / "a@b.c1").
_HAS_TLD = re.compile(r"\.[a-zA-Z]{2,}$")


def email_domain(email: str) -> str:
    """Return the lowercased domain part, or empty if missing."""
    email = (email or "").strip().lower()
    if "@" not in email:
        return ""
    return email.rsplit("@", 1)[-1].strip()


def is_disposable_email(email: str) -> bool:
    """True when the address uses a known disposable domain (or cheap fake pattern)."""
    email = (email or "").strip().lower()
    if "@" not in email:
        return False

    local, _, domain = email.partition("@")
    domain = domain.strip().rstrip(".")
    if not domain:
        return False

    # Strip one leading subdomain level for lists like mail.yopmail.com — only
    # when the registrable-looking suffix is in the set.
    if domain in DISPOSABLE_DOMAINS:
        return True
    parts = domain.split(".")
    if len(parts) > 2:
        parent = ".".join(parts[-2:])
        if parent in DISPOSABLE_DOMAINS:
            return True

    if _LONG_DIGIT_LOCAL.match(local):
        return True
    if not _HAS_TLD.search(domain):
        return True

    return False
