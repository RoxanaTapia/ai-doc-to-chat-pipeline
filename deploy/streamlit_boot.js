(() => {
  const boot = document.getElementById("app-boot");
  if (!boot) return;

  const BOOT_TEXT = "Loading the document…";
  const SLOW_TEXT = "Still starting up…";
  const SLOW_MS = 8000;
  const FAILSAFE_MS = 30000;

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

  const started = performance.now();

  const tick = () => {
    if (boot.dataset.done === "1") return;
    const elapsed = performance.now() - started;
    if (elapsed >= SLOW_MS && boot.textContent === BOOT_TEXT) {
      boot.textContent = SLOW_TEXT;
    }
    if (ready() || hasErrorSurface() || elapsed >= FAILSAFE_MS) {
      dismiss();
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
