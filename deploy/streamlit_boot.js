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

(() => {
  // Hide prior answer cards and show the busy status as soon as the visitor
  // asks again, without waiting for the next Streamlit round-trip.
  const BUSY_COPY = "Looking through the document";
  const PENDING = "app-ask-pending";
  const CARD_SEL = '[class*="st-key-answer_card_"], [class*="st-key-refusal_card_"]';
  const FAILSAFE_MS = 30000;
  let pendingSince = 0;

  const markPending = () => {
    document.body.classList.add(PENDING);
    pendingSince = performance.now();
    const status = document.querySelector(".app-status");
    if (status) {
      status.className = "app-status app-status--answering";
      status.textContent = BUSY_COPY;
    }
  };

  const clearPending = () => {
    document.body.classList.remove(PENDING);
    pendingSince = 0;
  };

  const isAskSubmit = (el) => {
    if (!el || !el.closest) return false;
    return !!el.closest('[class*="st-key-ask_submit"]');
  };

  const isChipButton = (el) => {
    if (!el || !el.closest) return false;
    return !!el.closest('[class*="st-key-chip_"]');
  };

  const onActivate = (event) => {
    const t = event.target;
    if (isAskSubmit(t) || isChipButton(t)) {
      markPending();
    }
  };

  const onKey = (event) => {
    if (event.key !== "Enter") return;
    const t = event.target;
    if (!t || t.tagName !== "INPUT") return;
    if (!t.closest || !t.closest('[data-testid="stTextInput"]')) return;
    markPending();
  };

  document.addEventListener("pointerdown", onActivate, true);
  document.addEventListener("keydown", onKey, true);

  const syncPending = () => {
    if (!document.body.classList.contains(PENDING)) return;
    // Drop the client hide once the server has cleared the card slot so the
    // idle run can paint new cards. Do not key off answering alone: we set
    // that class ourselves on click while old cards are still in the DOM.
    if (!document.querySelector(CARD_SEL)) {
      clearPending();
      return;
    }
    if (pendingSince && performance.now() - pendingSince >= FAILSAFE_MS) {
      clearPending();
    }
  };

  const obs = new MutationObserver(() => syncPending());
  const start = () => {
    obs.observe(document.body, { childList: true, subtree: true, attributes: true });
    syncPending();
  };
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
