(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);

  function show(el, data, status) {
    const ok = status >= 200 && status < 300;
    el.className = ok ? "ok" : "err";
    if (typeof data === "string") {
      el.textContent = "HTTP " + status + "\n" + data;
    } else {
      el.textContent = "HTTP " + status + "\n" + JSON.stringify(data, null, 2);
    }
  }

  async function api(method, path, body) {
    const opts = { method, headers: {} };
    if (body !== undefined) {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body);
    }
    const res = await fetch(path, opts);
    const text = await res.text();
    let json;
    try {
      json = JSON.parse(text);
    } catch (_) {
      json = text;
    }
    return { status: res.status, data: json };
  }

  function parseArray(text) {
    const v = JSON.parse(text);
    if (!Array.isArray(v)) {
      throw new Error("expected JSON array");
    }
    return v;
  }

  async function refreshHealth() {
    const badge = $("health-badge");
    try {
      const { status, data } = await api("GET", "/health");
      show($("health-out"), data, status);
      if (status === 200 && data && data.status === "ok") {
        badge.textContent = "ok — model: " + (data.model || "?") + ", predictions: " + (data.n_predictions ?? "?");
        badge.className = "ok";
      } else {
        badge.textContent = "unexpected health response";
        badge.className = "err";
      }
    } catch (e) {
      badge.textContent = "offline";
      badge.className = "err";
      $("health-out").textContent = String(e);
    }
  }

  $("btn-health").addEventListener("click", refreshHealth);

  $("btn-ready").addEventListener("click", async () => {
    const { status, data } = await api("GET", "/ready");
    show($("health-out"), data, status);
  });

  $("btn-predict").addEventListener("click", async () => {
    try {
      const input = parseArray($("predict-input").value);
      const body = {
        input,
        use_gh: $("predict-gh").checked,
        return_explanation: false,
      };
      const { status, data } = await api("POST", "/predict", body);
      show($("predict-out"), data, status);
    } catch (e) {
      show($("predict-out"), String(e), 0);
    }
  });

  $("btn-update").addEventListener("click", async () => {
    try {
      const input = parseArray($("update-input").value);
      const body = {
        input,
        correct_label: $("update-label").value,
        use_gh: $("update-gh").checked,
      };
      const { status, data } = await api("POST", "/update", body);
      show($("update-out"), data, status);
      if (status === 200) {
        refreshHealth();
      }
    } catch (e) {
      show($("update-out"), String(e), 0);
    }
  });

  $("btn-models").addEventListener("click", async () => {
    const q = $("models-summary").checked ? "?summary=true" : "";
    const { status, data } = await api("GET", "/models" + q);
    show($("models-out"), data, status);
  });

  $("btn-load").addEventListener("click", async () => {
    const name = $("load-name").value.trim();
    if (!name) {
      show($("models-out"), "name is required", 0);
      return;
    }
    const version = $("load-version").value.trim();
    const body = { name };
    if (version) {
      body.version = version;
    }
    const { status, data } = await api("POST", "/load", body);
    show($("models-out"), data, status);
    if (status === 200) {
      refreshHealth();
    }
  });

  $("btn-rng-get").addEventListener("click", async () => {
    const { status, data } = await api("GET", "/session/rng");
    show($("session-out"), data, status);
  });

  $("btn-rng-seed").addEventListener("click", async () => {
    const { status, data } = await api("POST", "/session/rng", { seed: 424242 });
    show($("session-out"), data, status);
  });

  refreshHealth();
})();
