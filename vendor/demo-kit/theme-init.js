/* Apply a saved theme before the first paint. Default is Auto.
   Read the rt_theme cookie first, then localStorage (same key as the home page). */
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

  function saved() {
    var fromCookie = readCookie(COOKIE);
    if (fromCookie === "dark" || fromCookie === "light" || fromCookie === "auto") return fromCookie;
    try {
      var stored = localStorage.getItem(STORAGE);
      if (stored === "dark" || stored === "light" || stored === "auto") return stored;
    } catch (e) {}
    return null;
  }

  function resolved(choice) {
    if (choice === "dark" || choice === "light") return choice;
    try {
      if (window.matchMedia("(prefers-color-scheme: dark)").matches) return "dark";
    } catch (e) {}
    return "light";
  }

  var stored = saved();
  var choice = stored === "dark" || stored === "light" || stored === "auto" ? stored : "auto";
  var theme = resolved(choice);
  document.documentElement.setAttribute("data-theme-choice", choice);
  document.documentElement.setAttribute("data-theme", theme);
  var meta = document.querySelector('meta[name="theme-color"]');
  if (meta) meta.setAttribute("content", theme === "dark" ? "#14061f" : "#f8f4f9");
})();
