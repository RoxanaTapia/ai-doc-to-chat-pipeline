# Security

## Background

How to report a vulnerability in private.

> **Takeaway:** Email the report. Keep exploit details out of GitHub issues, pull requests, and the live pilot.

## Report in private

Email [hello@roxanatapia.dev](mailto:hello@roxanatapia.dev) with:

- What is affected, and the core version (`vX.Y.Z`, or `ai_doc.__version__` on `main`)
- The impact, as far as you know it
- Steps to reproduce, with no secrets and no confidential PDFs
- A way to reply to you

That email is the private channel. A GitHub issue is public.

## After a report

The report stays private while a fix is prepared. A fix ships as a patch on the latest minor of the core, then a line in [CHANGELOG.md](CHANGELOG.md). Which versions receive fixes is in [SUPPORT.md](SUPPORT.md).

Say in the email if you want to be credited when the fix is released.
