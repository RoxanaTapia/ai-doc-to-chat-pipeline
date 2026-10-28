# Support

## Background

Which versions of the core receive fixes, and how to ask for help.

> **Takeaway:** Semantic versions apply to the `ai-doc` core only. The 1.x line gets fixes. Release tags are `vX.Y.Z`.

## What is versioned

| Piece | Version |
|-------|---------|
| Core package `ai-doc` (import `ai_doc`, code in `src/ai_doc`) | Semantic Versioning: `MAJOR.MINOR.PATCH` |
| Streamlit demo in `apps/demo` | No version of its own. It ships with the same tag as the core. |

The version in `pyproject.toml` and `ai_doc.__version__` is the core version.

| Bump | Meaning |
|------|---------|
| Major | Can break existing use of the core. |
| Minor | Adds behavior. Before 1.0.0, a minor bump can also break existing use. |
| Patch | A fix on the latest minor. |

## What gets fixes

Fixes ship on the **latest minor** of the core only. Older minors stay at the tag you already have.

**1.0.0** is the current supported release (tag `v1.0.0`). The 1.x line gets fixes.

Pin the `vX.Y.Z` tag you run when you need a snapshot that will not move.

## Release tags

- Release tags are `vX.Y.Z`. The current supported release is `v1.0.0`.
- Date-based deploy tags (`ai-doc-YYYY.MM.DD-N`) are no longer created.

## How to ask for help

Open an issue:

<https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline/issues/new/choose>

Include the core version (the `vX.Y.Z` tag, or `ai_doc.__version__`), what you expected, what happened, and the steps. Use a sample or non-confidential PDF. Leave secrets out of the issue.

Report a vulnerability in private. See [SECURITY.md](SECURITY.md).

[How I work with AI](docs/how-we-work.md)
