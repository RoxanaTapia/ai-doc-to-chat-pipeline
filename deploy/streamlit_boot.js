(() => {
  const boot = document.getElementById("app-boot");
  if (!boot) return;

  const ready = () => {
    const h1 = document.querySelector('[data-testid="stHeading"] h1, h1');
    const input = document.querySelector('[data-testid="stTextInput"] input');
    const status = document.querySelector(".app-status");
    if (!h1 || !input) return false;
    const h1Box = h1.getBoundingClientRect();
    const inputBox = input.getBoundingClientRect();
    return h1Box.height > 0 && inputBox.height > 0 && (!!status || inputBox.width > 0);
  };

  const dismiss = () => {
    if (boot.dataset.done === "1") return;
    boot.dataset.done = "1";
    window.setTimeout(() => boot.setAttribute("hidden", ""), 180);
  };

  const tick = () => {
    if (ready()) {
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
