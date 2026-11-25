/* Entry motion for demo pages. All of it stays off when reduced motion is requested.
   Count-up uses Intl.NumberFormat. The source sweep is the dk-sweep class. */
(function () {
  function reduced() {
    try {
      return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    } catch (e) {
      return false;
    }
  }

  function format(value) {
    var lang = document.documentElement.lang || "en";
    try {
      return new Intl.NumberFormat(lang).format(value);
    } catch (e) {
      return String(value);
    }
  }

  function countUp(el) {
    if (!el) return;
    var target = Number(el.getAttribute("data-count-up"));
    if (!isFinite(target)) return;
    var finalText = format(target);
    if (reduced()) {
      el.textContent = finalText;
      return;
    }
    var start = null;
    var ms = 800;
    function frame(now) {
      if (start === null) start = now;
      var t = Math.min(1, (now - start) / ms);
      var eased = 1 - Math.pow(1 - t, 3);
      el.textContent = format(Math.round(target * eased));
      if (t < 1) window.requestAnimationFrame(frame);
      else el.textContent = finalText;
    }
    window.requestAnimationFrame(frame);
  }

  function highlight(el) {
    if (!el || reduced()) return;
    el.classList.remove("dk-sweep");
    void el.offsetWidth;
    var done = function () {
      el.classList.remove("dk-sweep");
      el.removeEventListener("animationend", done);
    };
    el.addEventListener("animationend", done);
    el.classList.add("dk-sweep");
  }

  function boot() {
    Array.prototype.forEach.call(document.querySelectorAll("[data-count-up]"), countUp);
    Array.prototype.forEach.call(document.querySelectorAll(".dk-source"), highlight);
  }

  window.DemoKit = {
    countUp: countUp,
    highlight: highlight,
    reducedMotion: reduced,
  };

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
