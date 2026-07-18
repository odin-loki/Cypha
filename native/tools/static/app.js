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
      ["sequence_loaded", data.sequence_loaded ?? data.lm_loaded],
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

  function renderExperimentsList(data) {
    const root = $("experiments-list");
    if (!root) return;
    const models = data && Array.isArray(data.models) ? data.models : [];
    if (models.length === 0) {
      root.className = "empty";
      root.innerHTML =
        "No experiments in registry (start cypha_rest with <code>--registry</code>).";
      return;
    }
    const active = data.active_model;
    const rows = models
      .map((m) => {
        const name = m.name ?? "?";
        const ver = m.version ?? "?";
        const loaded = m.loaded ? "yes" : "no";
        const isActive = m.active || name + "/" + ver === active || name === active;
        return (
          "<tr><td>" +
          name +
          "</td><td>" +
          ver +
          "</td><td>" +
          loaded +
          "</td><td>" +
          (isActive ? "● active" : "—") +
          "</td></tr>"
        );
      })
      .join("");
    root.className = "";
    root.innerHTML =
      '<table class="exp-table"><thead><tr><th>name</th><th>version</th><th>loaded</th><th>status</th></tr></thead><tbody>' +
      rows +
      "</tbody></table>";
  }

  $("btn-models").addEventListener("click", async () => {
    const q = $("models-summary").checked ? "?summary=true" : "";
    const { status, data } = await api("GET", "/models" + q);
    show($("models-out"), data, status);
    if (status === 200) {
      renderExperimentsList(data);
    }
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

  async function lmGenerate(body) {
    return api("POST", "/generate", body);
  }

  async function lmGenerateStream(body, onChunk) {
    const payload = Object.assign({}, body, { stream: true });
    const opts = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    };
    const path = "/generate/stream";
    let res = await fetch(path, opts);
    if (res.status === 404 || !res.body) {
      return { streamed: false, result: await lmGenerate(body) };
    }
    if (!res.ok) {
      const text = await res.text();
      let data;
      try {
        data = JSON.parse(text);
      } catch (_) {
        data = text;
      }
      return { streamed: false, result: { status: res.status, data } };
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop() || "";
      for (const part of parts) {
        const line = part
          .split("\n")
          .map((l) => l.trim())
          .find((l) => l.startsWith("data: "));
        if (!line) continue;
        try {
          const chunk = JSON.parse(line.slice(6));
          onChunk(chunk);
        } catch (_) {
          /* ignore malformed SSE lines */
        }
      }
    }
    return { streamed: true, status: res.status };
  }

  const CHAT_VOCAB = 128;
  let chatContextText = "";
  let chatBusy = false;

  function encodePromptChars(text, vocabSize) {
    const uniq = [];
    for (const c of text) {
      if (!uniq.includes(c)) uniq.push(c);
    }
    uniq.sort();
    const limit = Math.max(vocabSize - 1, 1);
    if (uniq.length > limit) uniq.length = limit;
    const char2id = new Map();
    uniq.forEach((c, i) => char2id.set(c, i + 1));
    const ids = [];
    for (const c of text) {
      ids.push(char2id.has(c) ? char2id.get(c) : 0);
    }
    return ids;
  }

  function decodeGeneratedIds(ids, prompt, vocabSize) {
    const uniq = [];
    for (const c of prompt) {
      if (!uniq.includes(c)) uniq.push(c);
    }
    uniq.sort();
    const limit = Math.max(vocabSize - 1, 1);
    if (uniq.length > limit) uniq.length = limit;
    const id2char = new Map([[0, "?"]]);
    uniq.forEach((c, i) => id2char.set(i + 1, c));
    return ids.map((id) => id2char.get(id) ?? "?").join("");
  }

  function chatBody(promptIds) {
    const max_tokens = parseInt($("chat-max-tokens").value, 10) || 32;
    return {
      prompt_ids: promptIds,
      max_tokens,
      temperature: 0.9,
      strategy: $("chat-strategy").value,
      top_k: 40,
      top_p: 0.92,
      uncertainty_threshold: null,
      epistemic_halt: $("chat-epistemic").checked,
      stream: $("chat-stream").checked,
    };
  }

  function appendChatMessage(role, text, meta) {
    const log = $("chat-log");
    if (!log) return null;
    const el = document.createElement("div");
    el.className = "chat-msg " + role;
    const body = document.createElement("div");
    body.className = "chat-text";
    body.textContent = text;
    el.appendChild(body);
    if (meta) {
      const m = document.createElement("div");
      m.className = "meta";
      m.textContent = meta;
      el.appendChild(m);
    }
    log.appendChild(el);
    syncChatEmpty();
    log.scrollTop = log.scrollHeight;
    return el;
  }

  function syncChatEmpty() {
    const log = $("chat-log");
    const empty = $("chat-empty");
    if (!log || !empty) return;
    empty.classList.toggle("hidden", !!log.querySelector(".chat-msg"));
  }

  function setChatStatus(text, tone) {
    const el = $("chat-status");
    if (!el) return;
    el.textContent = text;
    if (tone) el.className = tone;
  }

  async function refreshChatReadiness() {
    try {
      const { status, data } = await api("GET", "/metrics");
      if (status !== 200 || !data) return;
      if (data.sequence_loaded ?? data.lm_loaded) {
        setChatStatus("Cypha sequence loaded — ready to chat.", "ok");
      } else {
        setChatStatus(
          "Sequence not loaded. Start cypha_rest with --sequence-checkpoint or POST /sequence/load.",
          "warn"
        );
      }
    } catch (_) {
      /* offline — keep default hint in HTML */
    }
  }

  function setChatBusy(busy) {
    chatBusy = busy;
    const send = $("btn-chat-send");
    const input = $("chat-input");
    if (send) send.disabled = busy;
    if (input) input.disabled = busy;
  }

  async function sendChatMessage() {
    if (chatBusy) return;
    const input = $("chat-input");
    const userText = input.value.trim();
    if (!userText) return;

    const promptText = chatContextText + userText;
    let promptIds;
    try {
      promptIds = encodePromptChars(promptText, CHAT_VOCAB);
    } catch (e) {
      appendChatMessage("system", "Encode error: " + String(e));
      return;
    }

    appendChatMessage("user", userText);
    input.value = "";
    chatContextText = promptText;

    const assistantEl = appendChatMessage("assistant", "…", "generating");
    assistantEl.classList.add("pending");
    const textNode = assistantEl.querySelector(".chat-text");
    const metaNode = assistantEl.querySelector(".meta");
    const generatedIds = [];
    let halted = false;
    let haltedEpistemic = false;
    const t0 = performance.now();
    setChatBusy(true);
    setChatStatus("Calling /generate…");

    try {
      const body = chatBody(promptIds);
      const useStream = body.stream;

      if (useStream) {
        const streamResult = await lmGenerateStream(body, (chunk) => {
          if (chunk.done) {
            if (chunk.halted_on_uncertainty) halted = true;
            return;
          }
          if (typeof chunk.token_id === "number") {
            generatedIds.push(chunk.token_id);
            const decoded = decodeGeneratedIds(generatedIds, promptText, CHAT_VOCAB);
            textNode.textContent = decoded || "…";
            if (metaNode) {
              metaNode.textContent =
                "token " +
                generatedIds.length +
                " · epistemic=" +
                (chunk.epistemic_var != null ? Number(chunk.epistemic_var).toFixed(4) : "?");
            }
          }
        });

        if (!streamResult.streamed) {
          const { status, data } = streamResult.result;
          if (status < 200 || status >= 300) {
            textNode.textContent = typeof data === "string" ? data : JSON.stringify(data, null, 2);
            assistantEl.classList.remove("pending");
            setChatStatus("Generate failed (HTTP " + status + ").");
            return;
          }
          const ids = Array.isArray(data.generated_ids) ? data.generated_ids : [];
          generatedIds.push(...ids);
          halted = !!data.halted_on_uncertainty;
          haltedEpistemic = !!data.halted_on_epistemic;
          textNode.textContent =
            decodeGeneratedIds(generatedIds, promptText, CHAT_VOCAB) ||
            (ids.length ? "[" + ids.join(", ") + "]" : "(empty)");
        }
      } else {
        const { status, data } = await lmGenerate(body);
        if (status < 200 || status >= 300) {
          textNode.textContent = typeof data === "string" ? data : JSON.stringify(data, null, 2);
          assistantEl.classList.remove("pending");
          setChatStatus("Generate failed (HTTP " + status + ").");
          return;
        }
        const ids = Array.isArray(data.generated_ids) ? data.generated_ids : [];
        generatedIds.push(...ids);
        halted = !!data.halted_on_uncertainty;
        haltedEpistemic = !!data.halted_on_epistemic;
        textNode.textContent =
          decodeGeneratedIds(generatedIds, promptText, CHAT_VOCAB) ||
          (ids.length ? "[" + ids.join(", ") + "]" : "(empty)");
      }

      assistantEl.classList.remove("pending");
      const decoded = textNode.textContent;
      chatContextText = promptText + decoded;
      const ms = (performance.now() - t0).toFixed(0);
      const haltNote = halted
        ? haltedEpistemic
          ? " · halted (epistemic)"
          : " · halted (uncertainty)"
        : "";
      if (metaNode) {
        metaNode.textContent =
          generatedIds.length +
          " tokens · " +
          ms +
          " ms" +
          haltNote +
          " · ids=[" +
          generatedIds.slice(0, 12).join(", ") +
          (generatedIds.length > 12 ? ", …" : "") +
          "]";
      }
      setChatStatus("Last reply: " + generatedIds.length + " tokens in " + ms + " ms.");
    } catch (e) {
      assistantEl.classList.remove("pending");
      textNode.textContent = String(e);
      setChatStatus("Request error.");
    } finally {
      setChatBusy(false);
    }
  }

  $("btn-chat-send").addEventListener("click", sendChatMessage);
  $("btn-chat-clear").addEventListener("click", () => {
    chatContextText = "";
    const log = $("chat-log");
    if (log) {
      log.innerHTML =
        '<div id="chat-empty" class="chat-empty">' +
        "<p>No messages yet</p>" +
        '<p class="hint">Type a prompt below. Sequence must be loaded via <code>--sequence-checkpoint</code> or <code>POST /sequence/load</code>.</p>' +
        "</div>";
    }
    refreshChatReadiness();
  });

  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (btn.dataset.tab === "lm") refreshChatReadiness();
    });
  });
  $("chat-input").addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && !ev.shiftKey) {
      ev.preventDefault();
      sendChatMessage();
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
      const { status, data } = await lmGenerate(body);
      show($("lm-out"), data, status);
    } catch (e) {
      show($("lm-out"), String(e), 0);
    }
  });

  $("btn-lm-metrics").addEventListener("click", async () => {
    const { status, data } = await api("GET", "/sequence/metrics");
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

  function renderIntelligenceSummary(data) {
    const root = $("intelligence-summary");
    if (!root || !data || typeof data !== "object") return;
    const items = [
      ["navigation_loss", data.navigation_loss],
      ["criticality_score", data.criticality_score_obs],
      ["source", data.source],
    ];
    root.innerHTML =
      '<div id="metrics-cards">' +
      items
        .map(
          ([lbl, val]) =>
            '<div class="metric-card"><div class="val">' +
            String(val ?? "—") +
            '</div><div class="lbl">' +
            lbl +
            "</div></div>"
        )
        .join("") +
      "</div>";
  }

  $("btn-intelligence-report").addEventListener("click", async () => {
    const { status, data } = await api("GET", "/intelligence/report");
    show($("intelligence-out"), data, status);
    if (status === 200) {
      renderIntelligenceSummary(data);
    }
  });

  drawLossChart();
  refreshHealth();
  refreshChatReadiness();
})();
