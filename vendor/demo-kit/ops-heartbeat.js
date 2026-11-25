/* Optional. Keeps an operator session alive.
   Does nothing unless <html data-operator="true">.
   After a real pointer or key press, while this tab is visible, POST /ops/ping.
   At most once every 5 minutes. A 401 sends the operator to the login page. */
(function () {
  if (document.documentElement.getAttribute("data-operator") !== "true") return;

  var GAP = 5 * 60 * 1000;
  var LOGIN = "/ops/login?expired=1";
  var last = 0;
  var timer = 0;
  var waiting = false;
  var stopped = false;

  function visible() {
    return document.visibilityState === "visible";
  }

  function ping() {
    if (stopped || !visible()) return;
    last = Date.now();
    waiting = false;
    if (timer) {
      window.clearTimeout(timer);
      timer = 0;
    }
    var done = function (res) {
      if (res && res.status === 401) {
        stopped = true;
        window.location.assign(LOGIN);
      }
    };
    try {
      window
        .fetch("/ops/ping", {
          method: "POST",
          credentials: "same-origin",
          cache: "no-store",
          headers: { Accept: "application/json" },
        })
        .then(done, function () {});
    } catch (e) {}
  }

  function onActivity(event) {
    if (stopped) return;
    if (!event || event.isTrusted === false) return;
    if (!visible()) return;
    var now = Date.now();
    if (now - last >= GAP) {
      ping();
      return;
    }
    if (waiting) return;
    waiting = true;
    timer = window.setTimeout(function () {
      timer = 0;
      waiting = false;
      if (visible()) ping();
    }, GAP - (now - last));
  }

  document.addEventListener("visibilitychange", function () {
    if (visible()) return;
    if (timer) {
      window.clearTimeout(timer);
      timer = 0;
    }
    waiting = false;
  });

  document.addEventListener("pointerdown", onActivity, true);
  document.addEventListener("keydown", onActivity, true);
})();
