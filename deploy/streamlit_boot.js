(() => {
  const boot = document.getElementById("app-boot");
  if (!boot) return;

  const BOOT_TEXT = "Loading the document…";
  const SLOW_TEXT = "Still starting up…";
  const SLOW_MS = 8000;
  const FAILSAFE_MS = 30000;
  // Theme CSS arrives via Streamlit markdown after the shell paints; bound the wait.
  const THEME_GATE_MS = 4000;

  const labelEl = () => boot.querySelector(".app-boot__label");

  const isVis = (el) => {
    if (!el) return false;
    const r = el.getBoundingClientRect();
    if (r.width <= 0 || r.height <= 0) return false;
    const cs = getComputedStyle(el);
    return cs.visibility !== "hidden" && cs.display !== "none" && parseFloat(cs.opacity || "1") > 0.01;
  };

  const ready = () => {
    const h1 = document.querySelector('[data-testid="stHeading"] h1, h1');
    const input = document.querySelector('[data-testid="stTextInput"] input');
    const status = document.querySelector(".app-status");
    if (!h1 || !input) return false;
    const h1Box = h1.getBoundingClientRect();
    const inputBox = input.getBoundingClientRect();
    return h1Box.height > 0 && inputBox.height > 0 && (!!status || inputBox.width > 0);
  };

  // Injected theme sets --app-bg; that means the stylesheet is in the DOM and applied.
  const themeApplied = () =>
    getComputedStyle(document.documentElement).getPropertyValue("--app-bg").trim() !== "";

  // Script exceptions, alerts, and the connection-error dialog must not stay covered.
  const hasErrorSurface = () => {
    const selectors = [
      '[data-testid="stException"]',
      '[data-testid="stAlert"]',
      '[data-testid="stNotification"]',
      '[data-testid="stConnectionStatus"]',
      '[data-testid="stModal"]',
      ".stException",
      "[class*='stException']",
    ];
    for (const sel of selectors) {
      const el = document.querySelector(sel);
      if (isVis(el)) return true;
    }
    const dialogs = document.querySelectorAll('[role="dialog"], [role="alertdialog"]');
    for (const el of dialogs) {
      if (!isVis(el)) continue;
      const text = (el.innerText || "").toLowerCase();
      if (
        text.includes("connection error")
        || text.includes("connecting")
        || text.includes("script execution error")
        || text.includes("exception")
      ) {
        return true;
      }
    }
    return false;
  };

  const dismiss = () => {
    if (boot.dataset.done === "1") return;
    boot.dataset.done = "1";
    window.setTimeout(() => boot.setAttribute("hidden", ""), 180);
  };

  // After the ask shell is up: theme applied (or timeout) → fonts.ready → one frame.
  const settleThemeThenDismiss = () => {
    const started = performance.now();
    let settling = false;

    const afterFonts = (next) => {
      const fonts = document.fonts;
      if (!fonts || !fonts.ready) {
        next();
        return;
      }
      let finished = false;
      const once = () => {
        if (finished) return;
        finished = true;
        next();
      };
      fonts.ready.then(once, once);
      window.setTimeout(once, Math.max(0, THEME_GATE_MS - (performance.now() - started)));
    };

    const poll = () => {
      if (boot.dataset.done === "1" || settling) return;
      if (hasErrorSurface()) {
        dismiss();
        return;
      }
      if (themeApplied() || performance.now() - started >= THEME_GATE_MS) {
        settling = true;
        afterFonts(() => {
          window.requestAnimationFrame(() => dismiss());
        });
        return;
      }
      window.requestAnimationFrame(poll);
    };

    // Prefer the load event when Streamlit (or us) ships theme as <link rel="stylesheet">.
    const themeLinks = document.querySelectorAll('link[rel="stylesheet"][data-app-theme], link[rel="stylesheet"][href*="theme"]');
    for (const link of themeLinks) {
      if (link.sheet) continue;
      link.addEventListener("load", () => poll(), { once: true });
      link.addEventListener("error", () => poll(), { once: true });
    }
    poll();
  };

  const started = performance.now();

  const tick = () => {
    if (boot.dataset.done === "1") return;
    const elapsed = performance.now() - started;
    const label = labelEl();
    if (elapsed >= SLOW_MS && label && label.textContent === BOOT_TEXT) {
      label.textContent = SLOW_TEXT;
    }
    if (hasErrorSurface() || elapsed >= FAILSAFE_MS) {
      dismiss();
      return;
    }
    if (ready()) {
      settleThemeThenDismiss();
      return;
    }
    window.requestAnimationFrame(tick);
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => window.requestAnimationFrame(tick));
  } else {
    window.requestAnimationFrame(tick);
  }
})();
