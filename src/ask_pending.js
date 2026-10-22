(() => {
  // Idempotent: Streamlit may re-emit st.html on every rerun.
  if (window.__appAskBootBound) return;
  window.__appAskBootBound = true;

  // On a real ask, hide prior cards and show the busy line via CSS on .app-status
  // (no separate busy node that can mount before status exists).
  const PENDING = "app-ask-pending";
  const DONE_SEL = "#app-ask-done";
  const testHooks = () => window.__APP_ASK_TEST_HOOKS === true;
  const FAILSAFE_MS = () => {
    if (testHooks()) {
      const override = Number(window.__APP_ASK_FAILSAFE_MS);
      if (Number.isFinite(override) && override > 0) return override;
    }
    return 30000;
  };

  let pendingBaseline = null;
  let failsafeTimer = null;

  const askInput = () => document.querySelector('[data-testid="stTextInput"] input');

  const askControl = (fromEl) => {
    const wrap = fromEl && fromEl.closest ? fromEl.closest('[class*="st-key-ask_submit"]') : null;
    if (!wrap) return null;
    return wrap.querySelector("button") || wrap.querySelector('[data-testid^="stBaseButton"]') || wrap;
  };

  const askEnabled = (fromEl) => {
    const control = askControl(fromEl) || document.querySelector(
      '[class*="st-key-ask_submit"] button, [class*="st-key-ask_submit"] [data-testid^="stBaseButton"]'
    );
    if (!control) return false;
    if (control.disabled) return false;
    if (control.getAttribute("aria-disabled") === "true") return false;
    if (control.hasAttribute("disabled")) return false;
    return true;
  };

  const trimmedAskValue = () => ((askInput() && askInput().value) || "").trim();

  const doneAskId = () => {
    const el = document.querySelector(DONE_SEL);
    return el ? el.getAttribute("data-ask-id") || "0" : "0";
  };

  const clearFailsafe = () => {
    if (failsafeTimer !== null) {
      window.clearTimeout(failsafeTimer);
      failsafeTimer = null;
    }
  };

  const clearPending = () => {
    document.body.classList.remove(PENDING);
    pendingBaseline = null;
    clearFailsafe();
  };

  const restoreFully = () => {
    clearPending();
  };

  const markPending = () => {
    document.body.classList.add(PENDING);
    pendingBaseline = doneAskId();
    clearFailsafe();
    failsafeTimer = window.setTimeout(() => {
      restoreFully();
    }, FAILSAFE_MS());
  };

  const isChipButton = (el) => {
    if (!el || !el.closest) return false;
    return !!el.closest('[class*="st-key-chip_"]');
  };

  const isAskSubmit = (el) => {
    if (!el || !el.closest) return false;
    return !!el.closest('[class*="st-key-ask_submit"]');
  };

  const onClick = (event) => {
    // Primary button only (ignore right-click / aux buttons).
    if (typeof event.button === "number" && event.button !== 0) return;
    if (event.defaultPrevented) return;
    const t = event.target;
    if (isChipButton(t)) {
      markPending();
      return;
    }
    if (!isAskSubmit(t)) return;
    if (!askEnabled(t)) return;
    if (!trimmedAskValue()) return;
    markPending();
  };

  const onKey = (event) => {
    if (event.key !== "Enter") return;
    if (event.isComposing || event.keyCode === 229) return;
    if (event.defaultPrevented) return;
    const t = event.target;
    if (!t || t.tagName !== "INPUT") return;
    if (!t.closest || !t.closest('[data-testid="stTextInput"]')) return;
    if (!askEnabled(t)) return;
    if (!trimmedAskValue()) return;
    markPending();
  };

  document.addEventListener("click", onClick, true);
  document.addEventListener("keydown", onKey, true);

  const syncPending = () => {
    if (!document.body.classList.contains(PENDING)) return;
    // Clear only when the server paints a new ask-done id (answer or refusal).
    // Do not key off "zero cards": fast refusals can swap cards in one mutation.
    if (pendingBaseline !== null && doneAskId() !== pendingBaseline) {
      clearPending();
    }
  };

  // Test-only hooks (enable with window.__APP_ASK_TEST_HOOKS = true before load).
  if (testHooks()) {
    window.__appAskPending = {
      markPending,
      clearPending,
      restoreFully,
      isPending: () => document.body.classList.contains(PENDING),
    };
  }

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
