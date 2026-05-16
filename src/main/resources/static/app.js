const state = {
  page: document.body.dataset.page,
  payload: null,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatValue(value) {
  if (value === null || value === undefined || value === "") return "-";
  return String(value);
}

function toast(message) {
  const node = $("#toast");
  if (!node) return;
  node.textContent = message;
  node.classList.add("visible");
  window.clearTimeout(node.dataset.timer);
  node.dataset.timer = window.setTimeout(() => node.classList.remove("visible"), 3200);
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || `Request failed with HTTP ${response.status}`);
  }
  return payload;
}

function markdownToHtml(markdown) {
  const lines = String(markdown || "").split(/\r?\n/);
  const html = [];
  let listType = null;

  function closeList() {
    if (listType) {
      html.push(`</${listType}>`);
      listType = null;
    }
  }

  function inline(text) {
    return escapeHtml(text)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/\*([^*]+)\*/g, "<em>$1</em>");
  }

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) {
      closeList();
      continue;
    }

    const heading = line.match(/^(#{1,3})\s+(.+)$/);
    if (heading) {
      closeList();
      html.push(`<h${heading[1].length}>${inline(heading[2])}</h${heading[1].length}>`);
      continue;
    }

    const numbered = line.match(/^\d+\.\s+(.+)$/);
    if (numbered) {
      if (listType !== "ol") {
        closeList();
        html.push("<ol>");
        listType = "ol";
      }
      html.push(`<li>${inline(numbered[1])}</li>`);
      continue;
    }

    const bullet = line.match(/^[-*]\s+(.+)$/);
    if (bullet) {
      if (listType !== "ul") {
        closeList();
        html.push("<ul>");
        listType = "ul";
      }
      html.push(`<li>${inline(bullet[1])}</li>`);
      continue;
    }

    closeList();
    html.push(`<p>${inline(line)}</p>`);
  }

  closeList();
  return html.join("");
}

function renderRuntime(runtime) {
  $$("[data-runtime]").forEach((node) => {
    const key = node.dataset.runtime;
    node.textContent = formatValue(runtime?.[key]);
  });
}

function renderReport(result) {
  const output = $("#report-output");
  if (!output) return;

  if (!result) {
    output.classList.add("empty");
    output.textContent = "No report yet. Ask the agent to compare the four parcel datasets.";
    renderTrace([]);
    return;
  }

  output.classList.toggle("empty", false);
  output.classList.toggle("error", Boolean(result.is_error));
  output.innerHTML = markdownToHtml(result.content);
  renderTrace(result.trace || []);
}

function renderTrace(trace) {
  const section = $("#trace-section");
  const output = $("#trace-output");
  if (!section || !output) return;

  section.classList.toggle("hidden", !trace.length);
  output.innerHTML = trace.map((step, index) => `
    <article class="trace-item">
      <h3>Step ${index + 1}</h3>
      <p><strong>Tool:</strong> <code>${escapeHtml(step.tool || "unknown")}</code></p>
      <p><strong>Arguments</strong></p>
      <pre class="code-block">${escapeHtml(JSON.stringify(step.arguments || {}, null, 2))}</pre>
      ${step.result_preview ? `<p><strong>Preview</strong></p><pre class="code-block">${escapeHtml(step.result_preview)}</pre>` : ""}
    </article>
  `).join("");
}

function renderCounts(counts) {
  $$("[data-count]").forEach((node) => {
    node.textContent = formatValue(counts?.[node.dataset.count]);
  });
}

function renderDeveloper(payload) {
  renderCounts(payload.counts);

  const preview = $("#latest-preview");
  if (preview) {
    preview.textContent = payload.latest_result?.content
      ? payload.latest_result.content.slice(0, 900)
      : "No result yet.";
  }

  const runtime = payload.runtime || {};
  $("#provider-field").value = runtime.provider || "huggingface";
  $("#model-field").value = runtime.model || "";
  $("#max-steps-field").value = runtime.max_steps || 6;
  $("#system-prompt-field").value = runtime.system_prompt || "";

  renderTelemetry(payload.latest_telemetry);
  renderTools(payload.mcp_tools || []);
  renderModels(payload.provider_models || []);
  renderRuntimeSummary(runtime);
}

function metric(label, value) {
  return `<div class="metric"><span>${escapeHtml(label)}</span><strong>${escapeHtml(formatValue(value))}</strong></div>`;
}

function renderTelemetry(telemetry) {
  const output = $("#telemetry-output");
  const raw = $("#raw-telemetry");
  if (!output) return;

  if (!telemetry) {
    output.innerHTML = metric("Status", "No model telemetry yet");
    if (raw) raw.textContent = "No telemetry yet.";
    return;
  }

  const usage = telemetry.usage || {};
  const rate = telemetry.rate_limit_headers || {};
  const rateError = telemetry.rate_limit_error || {};
  output.innerHTML = [
    metric("Provider", telemetry.provider),
    metric("Model", telemetry.model),
    metric("HTTP", telemetry.status_code),
    metric("Calls", telemetry.calls),
    metric("Prompt tokens", usage.prompt_tokens),
    metric("Completion tokens", usage.completion_tokens),
    metric("Total tokens", usage.total_tokens),
    metric("Queue time", usage.queue_time),
    metric("Token limit", rate["x-ratelimit-limit-tokens"]),
    metric("Tokens remaining", rate["x-ratelimit-remaining-tokens"]),
    metric("Used", rateError.used),
    metric("Requested", rateError.requested),
  ].join("");

  if (raw) raw.textContent = JSON.stringify(telemetry.raw || telemetry, null, 2);
}

function schemaType(schema) {
  if (!schema) return "any";
  if (schema.type === "array") return `array<${schema.items?.type || "item"}>`;
  return schema.type || "object";
}

function renderTools(tools) {
  const output = $("#tools-output");
  if (!output) return;

  if (!tools.length) {
    output.innerHTML = `<p class="tool-description">Refresh the MCP tool catalog to inspect the tools exposed to the agent.</p>`;
    return;
  }

  output.innerHTML = tools.map((tool) => {
    const schema = tool.inputSchema || tool.input_schema || {};
    const properties = schema.properties || {};
    const required = new Set(schema.required || []);
    const badges = Object.entries(properties).map(([name, prop]) => {
      const suffix = required.has(name) ? " *" : "";
      return `<span class="badge">${escapeHtml(`${name}: ${schemaType(prop)}${suffix}`)}</span>`;
    }).join("") || `<span class="badge">no arguments</span>`;

    return `
      <article class="tool-item">
        <h3>${escapeHtml(tool.name || "unnamed_tool")}</h3>
        <p class="tool-description">${escapeHtml((tool.description || "No description provided.").split("\\n\\n")[0])}</p>
        <div class="tool-badges">${badges}</div>
      </article>
    `;
  }).join("");
}

function renderModels(models) {
  const output = $("#models-output");
  if (!output) return;
  output.innerHTML = models.slice(0, 60).map((model) => `<span class="badge">${escapeHtml(model)}</span>`).join("");
}

function renderRuntimeSummary(runtime) {
  const output = $("#runtime-summary");
  if (!output) return;
  const rows = [
    ["Provider", runtime.provider],
    ["Model", runtime.model],
    ["Max steps", runtime.max_steps],
    ["Shared data path", runtime.data_root],
    ["MCP server", runtime.mcp_server],
    ["Hugging Face API", runtime.hf_api_base],
  ];
  output.innerHTML = rows.map(([label, value]) => `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(formatValue(value))}</dd></div>`).join("");
}

async function loadState() {
  state.payload = await requestJson("/api/state");
  renderRuntime(state.payload.runtime);
  if (state.page === "user") {
    renderReport(state.payload.latest_result);
  } else {
    renderDeveloper(state.payload);
  }
}

function setupUserPage() {
  const form = $("#prompt-form");
  const input = $("#prompt-input");
  const status = $("#run-status");
  if (!form) return;

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const prompt = input.value.trim();
    if (!prompt) {
      toast("Enter a request before sending it.");
      return;
    }

    const button = form.querySelector("button");
    button.disabled = true;
    status.textContent = "Running";
    status.classList.add("busy");

    try {
      state.payload = await requestJson("/api/agent", {
        method: "POST",
        body: JSON.stringify({ prompt }),
      });
      input.value = "";
      renderRuntime(state.payload.runtime);
      renderReport(state.payload.latest_result);
      toast("Analysis complete.");
    } catch (error) {
      toast(error.message);
    } finally {
      button.disabled = false;
      status.textContent = "Idle";
      status.classList.remove("busy");
    }
  });
}

function setupDeveloperPage() {
  const settingsButton = $("#save-settings");
  if (settingsButton) {
    settingsButton.addEventListener("click", async () => {
      settingsButton.disabled = true;
      try {
        state.payload = await requestJson("/api/settings", {
          method: "POST",
          body: JSON.stringify({
            provider: $("#provider-field").value,
            model: $("#model-field").value,
            max_steps: Number($("#max-steps-field").value),
            system_prompt: $("#system-prompt-field").value,
          }),
        });
        renderDeveloper(state.payload);
        toast("Settings saved.");
      } catch (error) {
        toast(error.message);
      } finally {
        settingsButton.disabled = false;
      }
    });
  }

  $$(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      $$(".tab").forEach((node) => node.classList.toggle("active", node === tab));
      $$("[data-panel]").forEach((panel) => {
        panel.classList.toggle("hidden", panel.dataset.panel !== tab.dataset.tab);
      });
    });
  });

  $("#refresh-tools")?.addEventListener("click", async (event) => {
    event.currentTarget.disabled = true;
    try {
      state.payload = await requestJson("/api/tools/refresh", { method: "POST", body: "{}" });
      renderDeveloper(state.payload);
      toast("MCP tools refreshed.");
    } catch (error) {
      toast(error.message);
    } finally {
      event.currentTarget.disabled = false;
    }
  });

  $("#refresh-models")?.addEventListener("click", async (event) => {
    event.currentTarget.disabled = true;
    try {
      state.payload = await requestJson("/api/models/refresh", { method: "POST", body: "{}" });
      renderDeveloper(state.payload);
      toast("Provider models loaded.");
    } catch (error) {
      toast(error.message);
    } finally {
      event.currentTarget.disabled = false;
    }
  });

  $$("[data-system-action]").forEach((button) => {
    button.addEventListener("click", async () => {
      button.disabled = true;
      try {
        state.payload = await requestJson("/api/system", {
          method: "POST",
          body: JSON.stringify({ action: button.dataset.systemAction }),
        });
        renderDeveloper(state.payload);
        toast("System action complete.");
      } catch (error) {
        toast(error.message);
      } finally {
        button.disabled = false;
      }
    });
  });
}

document.addEventListener("DOMContentLoaded", async () => {
  try {
    if (state.page === "user") setupUserPage();
    if (state.page === "developer") setupDeveloperPage();
    await loadState();
  } catch (error) {
    toast(error.message);
  }
});
