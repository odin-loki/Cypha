(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const lossHistory = [];
  let metricsPollTimer = null;

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

  function parseIntArray(text) {
    const v = parseArray(text);
    return v.map((x) => {
      const n = Number(x);
      if (!Number.isInteger(n)) {
        throw new Error("prompt_ids must be integers");
      }
      return n;
    });
  }

  function activateTab(name) {
    document.querySelectorAll(".tab-btn").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.tab === name);
    });
    document.querySelectorAll(".tab-panel").forEach((panel) => {
      panel.classList.toggle("active", panel.id === "tab-" + name);
    });
  }

  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => activateTab(btn.dataset.tab));
  });

  function drawLossChart() {
    const canvas = $("loss-chart");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    const pad = { l: 40, r: 12, t: 12, b: 28 };
    ctx.fillStyle = "#1a2332";
    ctx.fillRect(0, 0, w, h);

    if (lossHistory.length === 0) {
      ctx.fillStyle = "#8b9cb3";
      ctx.font = "13px system-ui,sans-serif";
      ctx.fillText("Run POST /update to record loss values", pad.l, h / 2);
      return;
    }

    const xs = lossHistory.map((_, i) => i);
    const ys = lossHistory.map((p) => p.loss);
    let yMin = Math.min(...ys);
    let yMax = Math.max(...ys);
    if (yMin === yMax) {
      yMin -= 0.5;
      yMax += 0.5;
    }
    const plotW = w - pad.l - pad.r;
    const plotH = h - pad.t - pad.b;

    ctx.strokeStyle = "#2d3a4f";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.l, pad.t);
    ctx.lineTo(pad.l, h - pad.b);
    ctx.lineTo(w - pad.r, h - pad.b);
    ctx.stroke();

    ctx.fillStyle = "#8b9cb3";
    ctx.font = "11px system-ui,sans-serif";
    ctx.fillText(yMax.toFixed(3), 4, pad.t + 10);
    ctx.fillText(yMin.toFixed(3), 4, h - pad.b);
    ctx.fillText("step", w / 2 - 16, h - 6);

    ctx.strokeStyle = "#3d8bfd";
    ctx.lineWidth = 2;
    ctx.beginPath();
    xs.forEach((xi, i) => {
      const x = pad.l + (xi / Math.max(xs.length - 1, 1)) * plotW;
      const y = pad.t + (1 - (ys[i] - yMin) / (yMax - yMin)) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    ctx.fillStyle = "#3dd68c";
    xs.forEach((xi, i) => {
      const x = pad.l + (xi / Math.max(xs.length - 1, 1)) * plotW;
      const y = pad.t + (1 - (ys[i] - yMin) / (yMax - yMin)) * plotH;
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  function recordLoss(data) {
    if (data && typeof data.loss === "number" && Number.isFinite(data.loss)) {
      lossHistory.push({ loss: data.loss, t: Date.now() });
      if (lossHistory.length > 200) lossHistory.shift();
      drawLossChart();
    }
  }

  function renderMetricsCards(data) {
    const root = $("metrics-cards");
    if (!root || !data || typeof data !== "object") return;
    const items = [
      ["uptime_s", data.uptime_seconds],
      ["predictions", data.n_predictions],
      ["corrections", data.n_corrections],
      ["models", data.registry_model_count],
      ["lm_loaded", data.lm_loaded],
      ["regression", data.regression_head_loaded],
    ];
    root.innerHTML = items
      .map(
        ([lbl, val]) =>
          '<div class="metric-card"><div class="val">' +
          String(val ?? "—") +
          '</div><div class="lbl">' +
          lbl +
          "</div></div>"
      )
      .join("");
  }

  async function refreshHealth() {
    const badge = $("health-badge");
    try {
      const { status, data } = await api("GET", "/health");
      show($("health-out"), data, status);
      if (status === 200 && data && data.status === "ok") {
        badge.textContent =
          "ok — model: " + (data.model || "?") + ", predictions: " + (data.n_predictions ?? "?");
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
        recordLoss(data);
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
    if (status === 200 && data) {
      const disp = $("session-rng-display");
      const pos = data.pos ?? "?";
      const stateLen = Array.isArray(data.state) ? data.state.length : 0;
      disp.textContent =
        "pos=" +
        pos +
        ", state[" +
        stateLen +
        "] — first 8: " +
        (Array.isArray(data.state) ? JSON.stringify(data.state.slice(0, 8)) : "[]");
    }
  });

  $("btn-rng-seed").addEventListener("click", async () => {
    const { status, data } = await api("POST", "/session/rng", { seed: 424242 });
    show($("session-out"), data, status);
    if (status === 200) {
      $("btn-rng-get").click();
    }
  });

  $("btn-lm-generate").addEventListener("click", async () => {
    try {
      const prompt_ids = parseIntArray($("lm-prompt").value);
      const max_tokens = parseInt($("lm-max-tokens").value, 10) || 8;
      const body = {
        prompt_ids,
        max_tokens,
        temperature: 1.0,
        strategy: "greedy",
        top_k: 0,
        top_p: 1.0,
        uncertainty_threshold: 1.0,
        stream: false,
      };
      const { status, data } = await api("POST", "/generate", body);
      show($("lm-out"), data, status);
    } catch (e) {
      show($("lm-out"), String(e), 0);
    }
  });

  $("btn-lm-metrics").addEventListener("click", async () => {
    const { status, data } = await api("GET", "/lm/metrics");
    show($("lm-out"), data, status);
  });

  async function fetchMetrics() {
    const { status, data } = await api("GET", "/metrics");
    show($("metrics-out"), data, status);
    if (status === 200) {
      renderMetricsCards(data);
    }
    return { status, data };
  }

  $("btn-metrics").addEventListener("click", fetchMetrics);

  $("btn-metrics-poll").addEventListener("click", () => {
    if (metricsPollTimer) {
      clearInterval(metricsPollTimer);
      metricsPollTimer = null;
      $("btn-metrics-poll").textContent = "Poll metrics";
      return;
    }
    fetchMetrics();
    metricsPollTimer = setInterval(fetchMetrics, 3000);
    $("btn-metrics-poll").textContent = "Stop polling";
  });

  drawLossChart();
  refreshHealth();
})();
