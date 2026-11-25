/* Theme menu. Auto follows this device. The choice is remembered here.
   Cookie rt_theme wins, then localStorage key "theme" (same key as the home page). */
(function () {
  var COOKIE = "rt_theme";
  var STORAGE = "theme";

  function readCookie(name) {
    var all = document.cookie ? document.cookie.split(";") : [];
    var i, item, eq, key, raw;
    for (i = 0; i < all.length; i++) {
      item = all[i];
      while (item.charAt(0) === " ") item = item.slice(1);
      eq = item.indexOf("=");
      key = eq === -1 ? item : item.slice(0, eq);
      if (key !== name) continue;
      raw = eq === -1 ? "" : item.slice(eq + 1);
      try {
        return decodeURIComponent(raw);
      } catch (e) {
        return raw;
      }
    }
    return null;
  }

  function portfolioDomain() {
    var host = location.hostname || "";
    var suffix = ".roxanatapia.dev";
    if (host === "roxanatapia.dev") return "; Domain=.roxanatapia.dev";
    if (host.length > suffix.length && host.slice(host.length - suffix.length) === suffix) {
      return "; Domain=.roxanatapia.dev";
    }
    return "";
  }

  function remember(picked) {
    try {
      localStorage.setItem(STORAGE, picked);
    } catch (e) {}
    var secure = location.protocol === "https:" ? "; Secure" : "";
    document.cookie =
      COOKIE +
      "=" +
      encodeURIComponent(picked) +
      "; Path=/; Max-Age=31536000; SameSite=Lax" +
      portfolioDomain() +
      secure;
  }

  function choice() {
    var current = document.documentElement.getAttribute("data-theme-choice");
    if (current === "light" || current === "dark" || current === "auto") return current;
    var fromCookie = readCookie(COOKIE);
    if (fromCookie === "light" || fromCookie === "dark" || fromCookie === "auto") return fromCookie;
    try {
      var stored = localStorage.getItem(STORAGE);
      if (stored === "light" || stored === "dark" || stored === "auto") return stored;
    } catch (e) {}
    return "auto";
  }

  function resolved(next) {
    if (next === "dark" || next === "light") return next;
    try {
      if (window.matchMedia("(prefers-color-scheme: dark)").matches) return "dark";
    } catch (e) {}
    return "light";
  }

  function labelFor(next) {
    if (next === "dark") return "Dark";
    if (next === "light") return "Light";
    return "Auto, same as this device";
  }

  function motionOk() {
    try {
      return !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    } catch (e) {
      return true;
    }
  }

  function paint(next, animate) {
    var picked = next === "light" || next === "dark" ? next : "auto";
    var apply = function () {
      var theme = resolved(picked);
      document.documentElement.setAttribute("data-theme-choice", picked);
      document.documentElement.setAttribute("data-theme", theme);
      remember(picked);
      var meta = document.querySelector('meta[name="theme-color"]');
      if (meta) meta.setAttribute("content", theme === "dark" ? "#14061f" : "#f8f4f9");
      var options = document.querySelectorAll("[data-set-theme]");
      Array.prototype.forEach.call(options, function (el) {
        el.setAttribute("aria-checked", el.getAttribute("data-set-theme") === picked ? "true" : "false");
      });
      var btn = document.getElementById("theme-toggle");
      if (btn) btn.setAttribute("aria-label", labelFor(picked));
    };
    if (animate && motionOk() && typeof document.startViewTransition === "function") {
      try {
        document.startViewTransition(apply);
        return;
      } catch (e) {}
    }
    apply();
  }

  function menu() {
    return document.getElementById("theme-menu");
  }

  function toggleBtn() {
    return document.getElementById("theme-toggle");
  }

  function options() {
    var list = menu();
    return list ? Array.prototype.slice.call(list.querySelectorAll("[data-set-theme]")) : [];
  }

  function openMenu() {
    var list = menu();
    var btn = toggleBtn();
    if (!list || !btn) return;
    list.hidden = false;
    btn.setAttribute("aria-expanded", "true");
    window.setTimeout(function () {
      if (list.hidden) return;
      var current = list.querySelector('[aria-checked="true"]') || list.querySelector("[data-set-theme]");
      if (current) current.focus();
    }, 0);
  }

  function closeMenu(focusBack) {
    var list = menu();
    var btn = toggleBtn();
    if (!list || !btn) return;
    list.hidden = true;
    btn.setAttribute("aria-expanded", "false");
    if (focusBack) btn.focus();
  }

  function syncExit() {
    var link = document.getElementById("demo-exit");
    if (!link) return;
    if (document.documentElement.getAttribute("data-operator") === "true") {
      link.setAttribute("href", "/ops/logout");
    }
  }

  function bind() {
    paint(choice(), false);
    syncExit();
    var btn = toggleBtn();
    if (!btn || btn.getAttribute("data-bound") === "true") return;
    btn.setAttribute("data-bound", "true");
    btn.addEventListener("click", function () {
      var list = menu();
      if (!list) return;
      if (list.hidden) openMenu();
      else closeMenu(false);
    });
    btn.addEventListener("keydown", function (event) {
      if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        openMenu();
      }
    });
    var list = menu();
    if (list) {
      list.addEventListener("keydown", function (event) {
        var items = options();
        var index = items.indexOf(document.activeElement);
        if (event.key === "Escape") {
          event.preventDefault();
          closeMenu(true);
        } else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
          event.preventDefault();
          if (!items.length) return;
          var step = event.key === "ArrowDown" ? 1 : -1;
          var next = index === -1 ? 0 : (index + step + items.length) % items.length;
          items[next].focus();
        } else if (event.key === "Home") {
          event.preventDefault();
          if (items[0]) items[0].focus();
        } else if (event.key === "End") {
          event.preventDefault();
          if (items.length) items[items.length - 1].focus();
        } else if (event.key === "Tab") {
          closeMenu(false);
        }
      });
      list.addEventListener("click", function (event) {
        var item = event.target.closest ? event.target.closest("[data-set-theme]") : null;
        if (!item) return;
        paint(item.getAttribute("data-set-theme"), true);
        closeMenu(true);
      });
    }
    document.addEventListener("click", function (event) {
      var list = menu();
      var button = toggleBtn();
      if (!list || list.hidden) return;
      if (list.contains(event.target) || (button && button.contains(event.target))) return;
      closeMenu(false);
    });
    document.addEventListener("keydown", function (event) {
      if (event.key !== "Escape") return;
      var list = menu();
      if (list && !list.hidden) closeMenu(true);
    });
    try {
      var media = window.matchMedia("(prefers-color-scheme: dark)");
      var onChange = function () {
        if (choice() === "auto") paint("auto", false);
      };
      if (media.addEventListener) media.addEventListener("change", onChange);
      else if (media.addListener) media.addListener(onChange);
    } catch (e) {}
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", bind);
  else bind();
})();
