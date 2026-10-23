(() => {
  // Idempotent: Streamlit may re-emit st.html on every rerun.
  if (window.__appAskBootBound) return;
  window.__appAskBootBound = true;

  // On a real ask, hide prior cards and show the busy line via CSS on .app-status
  // (no separate busy node that can mount before status exists).
  const PENDING = "app-ask-pending";
  const DONE_SEL = "#app-ask-done";
  const CARD_SEL = '[class*="st-key-answer_card_"], [class*="st-key-refusal_card_"]';
  const testHooks = () => window.__APP_ASK_TEST_HOOKS === true;
  const FAILSAFE_MS = () => {
    if (testHooks()) {
      const override = Number(window.__APP_ASK_FAILSAFE_MS);
      if (Number.isFinite(override) && override > 0) return override;
    }
    return 30000;
  };

  let pendingBaseline = null;
  let pendingCards = 0;
  let pendingKeys = [];
  let failsafeTimer = null;
  let clearRequest = 0;

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

  const readDoneId = () => {
    const el = document.querySelector(DONE_SEL);
    if (!el) return null;
    return el.getAttribute("data-ask-id") || "0";
  };

  const cardNodes = () => document.querySelectorAll(CARD_SEL);

  const cardKey = (el) => {
    const parts = String(el.className || "").split(/\s+/);
    for (const part of parts) {
      if (part.indexOf("st-key-answer_card_") === 0) return part;
      if (part.indexOf("st-key-refusal_card_") === 0) return part;
    }
    return "";
  };

  const clearFailsafe = () => {
    if (failsafeTimer !== null) {
      window.clearTimeout(failsafeTimer);
      failsafeTimer = null;
    }
  };

  const clearPending = () => {
    if (clearRequest) {
      window.cancelAnimationFrame(clearRequest);
      clearRequest = 0;
    }
    document.body.classList.remove(PENDING);
    pendingBaseline = null;
    pendingCards = 0;
    pendingKeys = [];
    clearFailsafe();
  };

  const restoreFully = () => {
    clearPending();
  };

  const markPending = () => {
    // A second Enter/click in the same ask must not resnapshot after cards unmount.
    if (!document.body.classList.contains(PENDING)) {
      const cards = cardNodes();
      pendingCards = cards.length;
      pendingKeys = [];
      for (const card of cards) {
        const key = cardKey(card);
        if (key) pendingKeys.push(key);
      }
      pendingBaseline = readDoneId();
    }
    document.body.classList.add(PENDING);
    clearFailsafe();
    failsafeTimer = window.setTimeout(() => {
      restoreFully();
    }, FAILSAFE_MS());
  };

  // New answer is on screen when the done id advanced, the new card is present,
  // every card from before the ask is back, and the typing line is gone.
  const answerReady = () => {
    if (!document.body.classList.contains(PENDING)) return false;
    const done = document.querySelector(DONE_SEL);
    if (!done) return false;
    const id = done.getAttribute("data-ask-id") || "0";
    if (pendingBaseline !== null && id === pendingBaseline) return false;
    if (document.querySelector(".app-typing")) return false;
    const cards = cardNodes();
    if (!(cards.length > pendingCards)) return false;
    const now = [];
    for (const card of cards) {
      const key = cardKey(card);
      if (key) now.push(key);
    }
    for (const key of pendingKeys) {
      if (now.indexOf(key) === -1) return false;
    }
    let fresh = false;
    for (const key of now) {
      if (pendingKeys.indexOf(key) === -1) fresh = true;
    }
    return fresh;
  };

  const scheduleClear = () => {
    if (clearRequest) return;
    clearRequest = window.requestAnimationFrame(() => {
      clearRequest = 0;
      if (!answerReady()) return;
      clearPending();
    });
  };

  const isChipButton = (el) => {
    if (!el || !el.closest) return false;
    return !!el.closest('[class*="st-key-chip_"]');
  };

  const isAskSubmit = (el) => {
    if (!el || !el.closest) return false;
    return !!el.closest('[class*="st-key-ask_submit"]');
  };

  const isAskInput = (el) => {
    if (!el || el.tagName !== "INPUT") return false;
    return !!(el.closest && el.closest('[data-testid="stTextInput"]'));
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

  const isEnterKey = (event) => event.key === "Enter" || event.key === "NumpadEnter";

  // Enter commits the text input on keypress, after keydown. Mark on keydown,
  // keypress, and keyup so a value that is only visible after keydown still counts.
  // Do not require the Ask button: it can be missing or disabled while Enter still submits.
  const onEnter = (event) => {
    if (!isEnterKey(event)) return;
    if (event.isComposing || event.keyCode === 229) return;
    const t = event.target;
    if (!isAskInput(t)) return;
    if (t.disabled || t.getAttribute("aria-disabled") === "true") return;
    if (!((t.value || "").trim())) return;
    markPending();
  };

  document.addEventListener("click", onClick, true);
  document.addEventListener("keydown", onEnter, true);
  document.addEventListener("keypress", onEnter, true);
  document.addEventListener("keyup", onEnter, true);

  const syncPending = () => {
    if (!answerReady()) return;
    // Drop the busy line in the same frame as the cards, not when the marker first changes.
    scheduleClear();
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
