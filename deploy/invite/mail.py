"""SMTP helpers for invite emails (stdlib smtplib)."""

from __future__ import annotations

import os
import smtplib
import ssl
from email.message import EmailMessage


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is not set")
    return value


def smtp_configured() -> bool:
    return bool(os.environ.get("SMTP_HOST", "").strip() and os.environ.get("SMTP_FROM", "").strip())


def send_email(
    *,
    to: str,
    subject: str,
    text_body: str,
    html_body: str,
    bcc: str | None = None,
) -> None:
    host = _require("SMTP_HOST")
    port = int(os.environ.get("SMTP_PORT", "587") or "587")
    from_addr = _require("SMTP_FROM")
    user = os.environ.get("SMTP_USER", "").strip()
    password = os.environ.get("SMTP_PASSWORD", "").strip()
    use_tls = os.environ.get("SMTP_TLS", "true").strip().lower() not in ("0", "false", "no")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to
    if bcc:
        msg["Bcc"] = bcc
    msg.set_content(text_body)
    msg.add_alternative(html_body, subtype="html")

    recipients = [to]
    if bcc:
        recipients.append(bcc)

    if use_tls and port == 465:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(host, port, context=context, timeout=30) as smtp:
            if user:
                smtp.login(user, password)
            smtp.send_message(msg, to_addrs=recipients)
        return

    with smtplib.SMTP(host, port, timeout=30) as smtp:
        smtp.ehlo()
        if use_tls:
            context = ssl.create_default_context()
            smtp.starttls(context=context)
            smtp.ehlo()
        if user:
            smtp.login(user, password)
        smtp.send_message(msg, to_addrs=recipients)
