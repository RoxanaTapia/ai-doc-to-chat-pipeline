(() => {
  // Idempotent: Streamlit may re-emit st.html on every rerun.
  if (window.__appAskBootBound) return;
  window.__appAskBootBound = true;

  // On a real ask, show the busy line via CSS on .app-status.
  // Earlier answer cards stay on screen. (No separate busy node.)
  const PENDING = "app-ask-pending";
  const askStarted = new Map();
  const askShown = new Map();
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
    pendingKeys = [];
    clearFailsafe();
  };

  const questionText = (fromEl) => {
    if (isChipButton(fromEl)) {
      const button = fromEl.closest("button") || fromEl;
      return ((button && button.innerText) || "").trim();
    }
    return trimmedAskValue();
  };

  const noteAskStart = (question) => {
    const text = (question || "").trim();
    if (!text || askShown.has(text)) return;
    askStarted.set(text, performance.now());
  };

  const formatAskSeconds = (ms) => {
    const seconds = Math.max(0, ms) / 1000;
    const shown = seconds >= 0.1 ? seconds.toFixed(1) : seconds.toFixed(2);
    return "Answered in " + shown + " s";
  };

  const paintBrowserTimes = () => {
    const nodes = document.querySelectorAll("[data-ask-timing]");
    nodes.forEach((el) => {
      const question = el.getAttribute("data-ask-question") || "";
      if (askShown.has(question)) {
        const saved = askShown.get(question);
        if (el.textContent !== saved) el.textContent = saved;
        return;
      }
      const started = askStarted.get(question);
      if (started == null) return;
      const card = el.closest(
        '[class*="st-key-answer_card_"], [class*="st-key-refusal_card_"]'
      );
      if (!card) return;
      const label = formatAskSeconds(performance.now() - started);
      askShown.set(question, label);
      askStarted.delete(question);
      if (el.textContent !== label) el.textContent = label;
    });
  };

  const scrollHighlightedPassage = () => {
    const anchors = document.querySelectorAll(".app-page-anchor");
    anchors.forEach((anchor) => {
      if (anchor.dataset.scrolled === "1") return;
      const ratio = parseFloat(anchor.getAttribute("data-highlight-top") || "");
      if (!Number.isFinite(ratio)) return;
      const block = anchor.closest('[data-testid="stVerticalBlock"]') || anchor.parentElement;
      const img = block ? block.querySelector("img") : null;
      if (!img) return;
      const box = img.getBoundingClientRect();
      if (!(box.height >= 20)) return;
      const y = box.top + box.height * ratio;
      const margin = 96;
      if (!(y >= margin) || y > window.innerHeight - 120) {
        window.scrollBy(0, y - Math.min(window.innerHeight * 0.33, 240));
      }
      anchor.dataset.scrolled = "1";
    });
  };

  const markPending = () => {
    // A repeat Enter/click in the same ask must not resnapshot or reset the failsafe.
    if (document.body.classList.contains(PENDING)) return;
    const cards = cardNodes();
    pendingKeys = [];
    for (const card of cards) {
      const key = cardKey(card);
      if (key) pendingKeys.push(key);
    }
    pendingBaseline = readDoneId();
    document.body.classList.add(PENDING);
    clearFailsafe();
    failsafeTimer = window.setTimeout(() => {
      clearPending();
    }, FAILSAFE_MS());
  };

  const historyCap = (done) => {
    const raw = Number(done.getAttribute("data-ask-cap"));
    if (Number.isFinite(raw) && raw > 0) return raw;
    return 0;
  };

  // Done id moved on, the card for that id is present, earlier cards are back,
  // and the typing line is gone. At the cap the oldest card is dropped.
  const answerReady = () => {
    if (!document.body.classList.contains(PENDING)) return false;
    const done = document.querySelector(DONE_SEL);
    if (!done) return false;
    const id = done.getAttribute("data-ask-id") || "0";
    if (pendingBaseline !== null && id === pendingBaseline) return false;
    if (document.querySelector(".app-typing")) return false;
    const now = [];
    for (const card of cardNodes()) {
      const key = cardKey(card);
      if (key) now.push(key);
    }
    const wantAnswer = "st-key-answer_card_" + id;
    const wantRefusal = "st-key-refusal_card_" + id;
    let fresh = false;
    for (const key of now) {
      if (key === wantAnswer || key === wantRefusal) fresh = true;
    }
    if (!fresh) return false;
    const cap = historyCap(done);
    const atCap = cap > 0 && pendingKeys.length >= cap;
    const oldest = pendingKeys.length ? pendingKeys[pendingKeys.length - 1] : "";
    for (const key of pendingKeys) {
      if (atCap && key === oldest) continue;
      if (now.indexOf(key) === -1) return false;
    }
    return true;
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
      noteAskStart(questionText(t));
      markPending();
      return;
    }
    if (!isAskSubmit(t)) return;
    if (!askEnabled(t)) return;
    if (!trimmedAskValue()) return;
    noteAskStart(trimmedAskValue());
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
    noteAskStart((t.value || "").trim());
    markPending();
  };

  document.addEventListener("click", onClick, true);
  document.addEventListener("keydown", onEnter, true);
  document.addEventListener("keypress", onEnter, true);
  document.addEventListener("keyup", onEnter, true);

  const syncPending = () => {
    paintBrowserTimes();
    scrollHighlightedPassage();
    if (!answerReady()) return;
    // Drop the busy line in the same frame as the cards, not when the marker first changes.
    scheduleClear();
  };

  // Test-only hooks (enable with window.__APP_ASK_TEST_HOOKS = true before load).
  if (testHooks()) {
    window.__appAskPending = {
      markPending,
      clearPending,
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
