# Changelog

## Background

Changes since [v0.9.0](https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline/releases/tag/v0.9.0) (2026-08-20). Each bullet under 1.0.0 is one merged pull request, newest first. `feat` is Added, `fix` is Fixed, and every other type is Changed.

Release tags are `vX.Y.Z`. Date-based deploy tags are no longer created. Existing ones stay.

> **Takeaway:** 1.0.0, tag `v1.0.0`, is the current release.

## [Unreleased]

No entries yet.

## [1.0.0] - 2026-10-28

The core is package `ai-doc` 1.0.0 (`ai_doc` under `src/ai_doc`). The Streamlit demo lives in `apps/demo`.

### Added

- feat(ui): open on the sample nda with the answer first (#167)
- feat(ui): one-click sample nda with cited answer cards (#165)
- feat(ui): make the pilot sidebar short and scannable (#152)

### Changed

- refactor(demo): move the Streamlit demo into apps/demo (#235)
- refactor(core): make the retrieval core an installable package (#234)
- build: add package metadata and a version (#232)
- docs: fix links to the removed pilot evaluation notes (#233)
- docs: add a short "How I work with AI" page (#226)
- Cleaned up the public docs (#189, #227, #228, #229)
- ci: run ruff in CI (#230)
- docs: update how-it-works image to say the demo uses a hosted model (#211)
- docs(pr): adopt concise PR and review standard (#191)
- docs(assets): add portfolio cover and how-it-works images (#173)
- refactor(rag): move section filters out of chunking.py (#161)
- refactor(ui): move the sources panels out of app.py (#160)
- refactor(ui): move session and upload helpers out of app.py (#159)
- refactor(ui): move page copy out of app.py (#158)
- docs(deploy): note that a reverse proxy in front handles HTTPS (#157)
- docs(agents): shorten agents.md to one screen (#156)
- chore(deps): drop requirements-dev and pre-commit config (#155)
- docs(readme): show the answer pipeline as one line (#154)
- chore: keep this repo to the product, and leave HTTPS with the reverse proxy in front (#153)
- chore(ui): retire the separate ui-only public demo (#148)
- docs: record v0.9.0 as the current release (#146)

### Fixed

- fix(ui): keep the busy line until the answer cards paint (#210)
- fix(ui): keep busy line until follow-up answer lands (#208)
- fix(ui): clear prior sources while a new answer loads (#206)
- fix(upload): show a plain error for unreadable PDFs (#204)
- fix(ui): keep boot overlay until theme styles apply (#203)
- fix(ui): stop chip flash and blank first paint (#199)
- fix(ui): restore subtitle, clear wrapped data note, fix refusal timing (#200)
- fix(ui): keep starter chips still on a fresh load (#195)
- fix(ui): keep the data note off the question box and tone down the subtitle (#193)
- fix(ui): keep header still when the data note opens (#187)
- fix(ui): hold starter questions until the document is ready (#186)
- fix(ui): answer typed questions on the first ask click (#185)
- fix(ui): accurate file-handling note and no starter flash (#179)
- fix(ui): one-click starters and the supporting page first (#177)
- fix(rag): answer the first question in a new session (#175)
- fix(ui): answer follow-up questions in the same session (#171)
- fix(ui): keep real answers and restore the demo buttons (#169)
- fix(rag): cite the right page for nda definition questions (#149)

## [0.9.0] - 2026-08-20

Tagged release. This log starts at this tag.

[Unreleased]: https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline/compare/v0.9.0...HEAD
[1.0.0]: https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline/compare/v0.9.0...v1.0.0
[0.9.0]: https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline/releases/tag/v0.9.0
