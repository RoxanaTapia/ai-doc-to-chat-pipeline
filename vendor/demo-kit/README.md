# Demo kit

Shared look for the portfolio demos. Copy this folder into a demo repo. Do not load any file from another origin, so the demo needs no CSP change.

`VERSION` is the pin. It is `1` until the contract of this folder changes. `SHA256SUMS` is the byte check, the same idea as `scripts/sync-shared.sh` plus `cmp` in CI: the demo fails if a copied file was edited or left out.

## Files

| File | Job |
|------|-----|
| `demo-kit.css` | Light and dark tokens, type, 18px cards, pill buttons, header, motion |
| `theme-init.js` | Sets `data-theme` before first paint |
| `theme.js` | Auto / Light / Dark menu |
| `motion.js` | Count-up and the source highlight. Optional |
| `ops-heartbeat.js` | Operator session ping. Optional. Quiet unless the operator flag is set |
| `preview.html` | Header, buttons, card, and chips in either theme |
| `favicon.svg`, `favicon-32.png`, `apple-touch-icon.png` | Same mark as the home page |
| `fonts/` | Plus Jakarta Sans and Bricolage Grotesque, with the OFL text |

Font URLs in the CSS are relative to `demo-kit.css`, so they keep working wherever you put the folder.

## Header

The bar is 56px, sticky, translucent, with a blur. On the left: the framed R, the RoxanaTapia.dev wordmark, a divider, and the product-name slot (`.dk-product`). On the right: the theme menu and one pill. The pill reads Exit demo. Set `data-operator="true"` on `<html>` and it reads Log out, and `theme.js` points it at `/ops/logout`.

Leave `.dk-product` empty (no spaces) when a page has no product name. The divider hides with it.

Under the bar, one thin line: Portfolio demo · fictional business · sample data.

The footer slot is `.dk-footer-slot`. The preview fills it with Built by Roxana Tapia, How it's built, and Book a call. Replace those links in the demo. Point How it's built at that demo's own notes.

Paths below are for a page that is not inside this folder. `preview.html` uses bare filenames because it sits next to the files.

```html
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="theme-color" content="#f8f4f9" />
    <link rel="icon" href="/demo-kit/favicon.svg" type="image/svg+xml" />
    <link rel="icon" href="/demo-kit/favicon-32.png" type="image/png" sizes="32x32" />
    <link rel="apple-touch-icon" href="/demo-kit/apple-touch-icon.png" />
    <script src="/demo-kit/theme-init.js"></script>
    <link rel="stylesheet" href="/demo-kit/demo-kit.css" />
  </head>
  <body>
    <header class="dk-header">
      <div class="dk-bar">
        <div class="dk-brand">
          <a class="dk-wordmark" href="https://roxanatapia.dev/">
            <img class="dk-mark" src="/demo-kit/favicon.svg" alt="" width="16" height="16" />
            <span class="dk-wordmark-name">RoxanaTapia<span class="dk-wordmark-dev">.dev</span></span>
          </a>
          <span class="dk-divider" aria-hidden="true"></span>
          <span class="dk-product">Product name</span>
        </div>
        <div class="dk-tools">
          <div class="theme-menu">
            <button type="button" class="theme-toggle" id="theme-toggle" aria-haspopup="menu" aria-expanded="false" aria-controls="theme-menu" aria-label="Auto">
              <svg class="theme-icon theme-icon-auto" viewBox="0 0 24 24" width="20" height="20" aria-hidden="true" focusable="false"><rect x="3.5" y="4.5" width="17" height="11" rx="1.6" fill="none" stroke="currentColor" stroke-width="1.7"/><path d="M8 20h8M12 15.5V20" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/></svg>
              <svg class="theme-icon theme-icon-light" viewBox="0 0 24 24" width="20" height="20" aria-hidden="true" focusable="false"><circle cx="12" cy="12" r="3.5" fill="none" stroke="currentColor" stroke-width="1.7"/><path d="M12 3v2.2M12 18.8V21M3 12h2.2M18.8 12H21M5.4 5.4l1.6 1.6M17 17l1.6 1.6M18.6 5.4 17 7M7 17l-1.6 1.6" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/></svg>
              <svg class="theme-icon theme-icon-dark" viewBox="0 0 24 24" width="20" height="20" aria-hidden="true" focusable="false"><path d="M20 14.6A7.4 7.4 0 0 1 9.4 4 7.2 7.2 0 1 0 20 14.6Z" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"/></svg>
            </button>
            <div class="theme-dropdown" id="theme-menu" role="menu" aria-labelledby="theme-toggle" hidden>
              <button type="button" class="theme-option" role="menuitemradio" data-set-theme="auto" aria-checked="true">
                <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true" focusable="false"><rect x="3.5" y="4.5" width="17" height="11" rx="1.6" fill="none" stroke="currentColor" stroke-width="1.7"/><path d="M8 20h8M12 15.5V20" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/></svg>
                <span class="theme-option-copy">
                  <span>Auto</span>
                  <span class="theme-hint">Same as this device</span>
                </span>
              </button>
              <button type="button" class="theme-option" role="menuitemradio" data-set-theme="light" aria-checked="false">
                <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true" focusable="false"><circle cx="12" cy="12" r="3.5" fill="none" stroke="currentColor" stroke-width="1.7"/><path d="M12 3v2.2M12 18.8V21M3 12h2.2M18.8 12H21M5.4 5.4l1.6 1.6M17 17l1.6 1.6M18.6 5.4 17 7M7 17l-1.6 1.6" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/></svg>
                <span>Light</span>
              </button>
              <button type="button" class="theme-option" role="menuitemradio" data-set-theme="dark" aria-checked="false">
                <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true" focusable="false"><path d="M20 14.6A7.4 7.4 0 0 1 9.4 4 7.2 7.2 0 1 0 20 14.6Z" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"/></svg>
                <span>Dark</span>
              </button>
            </div>
          </div>
          <a class="btn btn-primary dk-exit" id="demo-exit" href="https://roxanatapia.dev/">
            <span class="dk-exit-visitor">Exit demo</span>
            <span class="dk-exit-ops">Log out</span>
          </a>
        </div>
      </div>
      <p class="dk-note">Portfolio demo · fictional business · sample data</p>
    </header>
    <footer class="dk-footer">
      <div class="dk-footer-slot">
        <a href="https://roxanatapia.dev/">Built by Roxana Tapia</a>
        <span aria-hidden="true">·</span>
        <a href="https://github.com/RoxanaTapia">How it's built</a>
        <span aria-hidden="true">·</span>
        <a href="https://calendar.google.com/calendar/appointments/schedules/AcZssZ3r5LJ7bPLsMMNRH2fcH9DkzDNTf8oA8XIg0NX9aNpoKezPGMN2Z2OTD-baqG9XoB5vfehzJnzt?gv=true">Book a call</a>
      </div>
    </footer>
    <script src="/demo-kit/theme.js"></script>
  </body>
</html>
```

For an operator session, add the flag on the root element: `<html lang="en" data-operator="true">`. Include `ops-heartbeat.js` on that page. It POSTs `/ops/ping` only after a pointer or key event while the tab is visible, at most every 5 minutes, and goes to `/ops/login?expired=1` on a 401. Without the flag the script does nothing.

## Theme

`theme-init.js` reads the `rt_theme` cookie first, then `localStorage` key `theme`, then Auto. Auto follows `prefers-color-scheme`. It sets `data-theme` and `theme-color` before first paint (`#f8f4f9` light, `#14061f` dark).

`theme.js` is the same menu as the home page: one icon, then Auto, Light, and Dark. A choice is written back to the cookie and to `localStorage`. On `roxanatapia.dev` and its subdomains the cookie uses `Domain=.roxanatapia.dev` so those hosts can share it. Other hosts keep a host-only cookie.

## Motion

All of this is off under `prefers-reduced-motion: reduce`.

- `.dk-enter` fades in with `@starting-style`.
- `.dk-source` gets one highlight sweep when `motion.js` loads. Call `DemoKit.highlight(element)` to run it again.
- `[data-count-up="12480"]` counts up with `Intl.NumberFormat` for the page language.
- Choosing a theme uses a short view transition when the browser has one.

## Vendor

Copy `sites/shared/demo-kit/` into the demo repo. Keep the folder together. Record the git SHA you copied and the `VERSION` string.

In the demo CI, from the copied folder:

```bash
sha256sum --quiet -c SHA256SUMS
test "$(cat VERSION)" = "1"
```

`check-hash.sh` runs the sum check. That is the demo-side match for `scripts/sync-shared.sh`: this repo owns the canonical files, the demo proves its copy still matches.

After you change the kit in this repo, refresh the sums from the repo root, then commit `SHA256SUMS` with the change. Bump `VERSION` when the contract changes (new required markup, a renamed hook, a different cookie).

```bash
cd sites/shared/demo-kit
find . -type f ! -name SHA256SUMS -print | LC_ALL=C sort | sed 's|^\./||' > /tmp/demo-kit-files.txt
: > /tmp/demo-kit-sums.txt
while IFS= read -r file; do
  sha256sum "$file" >> /tmp/demo-kit-sums.txt
done < /tmp/demo-kit-files.txt
mv /tmp/demo-kit-sums.txt SHA256SUMS
```

## Preview

From the repo root:

```bash
python3 -m http.server 8765
```

Open `http://127.0.0.1:8765/sites/shared/demo-kit/preview.html`. Use the theme menu for light and dark.
