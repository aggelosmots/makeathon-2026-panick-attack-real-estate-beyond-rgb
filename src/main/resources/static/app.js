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

function filenameForFormat(format) {
  const normalized = String(format || "md").toLowerCase();
  return `real-estate-beyond-rgb-report.${normalized}`;
}

function filenameFromDisposition(disposition, fallback) {
  const match = String(disposition || "").match(/filename="?([^"]+)"?/i);
  return match ? match[1] : fallback;
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
  let paragraph = [];
  let inFence = false;
  let fenceLines = [];

  function closeList() {
    if (listType) {
      html.push(`</${listType}>`);
      listType = null;
    }
  }

  function closeParagraph() {
    if (paragraph.length) {
      html.push(`<p>${inline(paragraph.join(" "))}</p>`);
      paragraph = [];
    }
  }

  function inline(text) {
    return escapeHtml(text)
      .replace(/!\[([^\]]*)\]\(([^)]+)\)/g, '<img class="markdown-image" src="$2" alt="$1">')
      .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>')
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/\*([^*]+)\*/g, "<em>$1</em>");
  }

  function isTableSeparator(line) {
    return /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(line);
  }

  function tableCells(line) {
    return line
      .trim()
      .replace(/^\|/, "")
      .replace(/\|$/, "")
      .split("|")
      .map((cell) => cell.trim());
  }

  for (let index = 0; index < lines.length; index += 1) {
    const rawLine = lines[index];
    const line = rawLine.trim();
    const nextLine = lines[index + 1]?.trim() || "";

    if (line.startsWith("```")) {
      closeParagraph();
      closeList();
      if (inFence) {
        html.push(`<pre class="code-block"><code>${escapeHtml(fenceLines.join("\n"))}</code></pre>`);
        fenceLines = [];
        inFence = false;
      } else {
        inFence = true;
      }
      continue;
    }

    if (inFence) {
      fenceLines.push(rawLine);
      continue;
    }

    if (!line) {
      closeParagraph();
      closeList();
      continue;
    }

    const heading = line.match(/^(#{1,3})\s+(.+)$/);
    if (heading) {
      closeParagraph();
      closeList();
      html.push(`<h${heading[1].length}>${inline(heading[2])}</h${heading[1].length}>`);
      continue;
    }

    if (line.includes("|") && isTableSeparator(nextLine)) {
      closeParagraph();
      closeList();
      const headers = tableCells(line);
      index += 1;
      const rows = [];
      while (index + 1 < lines.length && lines[index + 1].trim().includes("|")) {
        index += 1;
        rows.push(tableCells(lines[index]));
      }
      html.push(`
        <div class="markdown-table-wrap">
          <table class="markdown-table">
            <thead><tr>${headers.map((cell) => `<th>${inline(cell)}</th>`).join("")}</tr></thead>
            <tbody>
              ${rows.map((row) => `<tr>${headers.map((_, cellIndex) => `<td>${inline(row[cellIndex] || "")}</td>`).join("")}</tr>`).join("")}
            </tbody>
          </table>
        </div>
      `);
      continue;
    }

    const numbered = line.match(/^\d+\.\s+(.+)$/);
    if (numbered) {
      closeParagraph();
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
      closeParagraph();
      if (listType !== "ul") {
        closeList();
        html.push("<ul>");
        listType = "ul";
      }
      html.push(`<li>${inline(bullet[1])}</li>`);
      continue;
    }

    closeList();
    paragraph.push(line);
  }

  if (inFence) {
    html.push(`<pre class="code-block"><code>${escapeHtml(fenceLines.join("\n"))}</code></pre>`);
  }
  closeParagraph();
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
    output.textContent = "No report yet.";
    renderFigures(null);
    setDownloadsEnabled(false);
    renderTrace([]);
    return;
  }

  output.classList.toggle("empty", false);
  output.classList.toggle("error", Boolean(result.is_error));
  output.innerHTML = markdownToHtml(result.content);
  renderFigures(result);
  setDownloadsEnabled(!result.is_error);
  renderTrace(result.trace || []);
}

function setDownloadsEnabled(enabled) {
  $$(".download-button").forEach((button) => {
    button.disabled = !enabled;
  });
}

function renderFigures(result) {
  const output = $("#figures-output");
  if (!output) return;

  const metrics = extractMetrics(result?.content || "");
  const plots = extractPlotOutputs(result);
  if (!metrics.length && !plots.length) {
    output.classList.add("hidden");
    output.innerHTML = "";
    return;
  }

  const rows = [
    ["Mean NDVI", "ndvi"],
    ["Healthy Coverage", "coverage"],
    ["Uniformity", "uniformity"],
  ];
  output.classList.remove("hidden");
  output.innerHTML = [
    ...plots.map(plotCard),
    ...rows.map(([title, key]) => figureCard(title, metrics, key)),
  ].join("");
}

function extractPlotOutputs(result) {
  const plots = [];
  const seen = new Set();

  function addPlot(item) {
    if (!item || item.success === false || !item.relative_path) return;
    const path = String(item.relative_path);
    const isPlotOutput = item.plot_name || item.chart_type || path.startsWith("plots/");
    if (!isPlotOutput) return;
    if (seen.has(path)) return;
    seen.add(path);
    plots.push({
      path,
      title: item.plot_name || item.title || path.split("/").pop(),
      type: item.chart_type || path.split(".").pop(),
    });
  }

  function visit(value) {
    if (!value || typeof value !== "object") return;
    if (Array.isArray(value)) {
      value.forEach(visit);
      return;
    }
    addPlot(value);
    if (Array.isArray(value.plots)) value.plots.forEach(visit);
  }

  for (const step of result?.trace || []) {
    const preview = step.result_preview;
    if (!preview) continue;
    try {
      visit(JSON.parse(preview));
    } catch {
      // Non-JSON tool previews are ignored by the figure renderer.
    }
  }

  return plots;
}

function plotCard(plot) {
  const url = `/api/data?path=${encodeURIComponent(plot.path)}`;
  const title = String(plot.title || "Plot").replaceAll("_", " ");
  const isImage = ["svg", "png"].includes(String(plot.type || "").toLowerCase());
  return `
    <article class="figure-card plot-card">
      <h3>${escapeHtml(title)}</h3>
      ${isImage
        ? `<img class="plot-image" src="${escapeHtml(url)}" alt="${escapeHtml(title)}">`
        : `<a class="plot-link" href="${escapeHtml(url)}" target="_blank" rel="noreferrer">${escapeHtml(plot.path)}</a>`}
    </article>
  `;
}

function figureCard(title, metrics, key) {
  const values = metrics.filter((item) => Number.isFinite(item[key]));
  if (!values.length) return "";
  const max = Math.max(...values.map((item) => Math.abs(item[key])), 1);
  return `
    <article class="figure-card">
      <h3>${escapeHtml(title)}</h3>
      <div class="bar-list">
        ${values.map((item) => {
          const width = Math.max(4, Math.min(100, Math.abs(item[key]) / max * 100));
          return `
            <div class="bar-row">
              <span>${escapeHtml(item.name)}</span>
              <div class="bar-track"><div class="bar-fill" style="width: ${width}%"></div></div>
              <strong>${escapeHtml(formatMetric(item[key], key))}</strong>
            </div>
          `;
        }).join("")}
      </div>
    </article>
  `;
}

function formatMetric(value, key) {
  if (!Number.isFinite(value)) return "-";
  if (key === "coverage") return `${value.toFixed(1)}%`;
  return value.toFixed(3);
}

function extractMetrics(content) {
  const names = ["Arkadia_2", "Arkadia 2", "Arkadia", "Magnisia", "Veroia"];
  const byName = new Map();
  String(content).split(/\r?\n/).forEach((line) => {
    const normalized = line.replace(/\|/g, " ");
    const matchedName = names.find((name) => new RegExp(`\\b${name.replace(" ", "[ _-]?")}\\b`, "i").test(normalized));
    if (!matchedName) return;

    const key = matchedName.toLowerCase().replace(/\s+/g, "_");
    const item = byName.get(key) || { name: matchedName.replace("_", " ") };
    const lower = normalized.toLowerCase();
    const numbers = normalized.match(/-?\d+(?:\.\d+)?%?/g) || [];
    const numericValues = numbers.map((value) => Number(value.replace("%", ""))).filter(Number.isFinite);

    if (lower.includes("ndvi") && !lower.includes("standard") && item.ndvi === undefined) {
      item.ndvi = numericValues.find((value) => value >= -1 && value <= 1);
    }
    if ((lower.includes("coverage") || lower.includes("healthy")) && item.coverage === undefined) {
      item.coverage = numericValues.find((value) => value >= 0 && value <= 100);
    }
    if ((lower.includes("standard deviation") || lower.includes("uniformity") || lower.includes("std")) && item.uniformity === undefined) {
      const std = numericValues.find((value) => value >= 0 && value <= 1);
      if (std !== undefined) item.uniformity = Math.max(0, 1 - std);
    }
    if (item.ndvi === undefined && item.coverage === undefined && item.uniformity === undefined && numericValues.length >= 3) {
      const ndvi = numericValues.find((value) => value >= -1 && value <= 1);
      const std = numericValues.find((value, index) => index > 0 && value >= 0 && value <= 1);
      const coverage = numericValues.find((value) => value > 1 && value <= 100);
      if (ndvi !== undefined) item.ndvi = ndvi;
      if (std !== undefined) item.uniformity = Math.max(0, 1 - std);
      if (coverage !== undefined) item.coverage = coverage;
    }

    if (item.ndvi !== undefined || item.coverage !== undefined || item.uniformity !== undefined) {
      byName.set(key, item);
    }
  });
  return Array.from(byName.values());
}

function renderTrace(trace) {
  const section = $("#trace-section");
  const output = $("#trace-output");
  if (!section || !output) return;

  section.classList.toggle("hidden", !trace.length);
  output.innerHTML = trace.map((step, index) => `
    <article class="trace-item">
      <h3>${escapeHtml(step.tool || `Step ${index + 1}`)}</h3>
      <p><strong>Arguments</strong></p>
      <pre class="code-block">${escapeHtml(JSON.stringify(step.arguments || {}, null, 2))}</pre>
      ${step.result_preview ? `<p><strong>Data</strong></p><pre class="code-block">${escapeHtml(step.result_preview)}</pre>` : ""}
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
    await loadDatasets();
  } else {
    renderDeveloper(state.payload);
  }
}

async function loadDatasets() {
  const select = $("#dataset-select");
  if (!select) return;

  try {
    const payload = await requestJson("/api/datasets");
    const datasets = payload.datasets || [];
    select.innerHTML = datasets.map((item) => `
      <option value="${escapeHtml(item.relative_path)}" selected>${escapeHtml(item.name)}</option>
    `).join("");
  } catch (error) {
    select.innerHTML = `<option value="" disabled>${escapeHtml(error.message)}</option>`;
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
    const datasets = Array.from($("#dataset-select")?.selectedOptions || [])
      .map((option) => option.value)
      .filter(Boolean);
    button.disabled = true;
    status.textContent = "Running";
    status.classList.add("busy");

    try {
      state.payload = await requestJson("/api/agent", {
        method: "POST",
        body: JSON.stringify({ prompt, datasets }),
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

  $$(".download-button").forEach((button) => {
    button.addEventListener("click", async () => {
      const format = button.dataset.downloadFormat;
      if (!format) return;
      button.disabled = true;
      try {
        const response = await fetch(`/api/download?format=${encodeURIComponent(format)}`);
        if (!response.ok) {
          const payload = await response.json().catch(() => ({}));
          throw new Error(payload.error || `Download failed with HTTP ${response.status}`);
        }
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = filenameFromDisposition(
          response.headers.get("Content-Disposition"),
          filenameForFormat(format),
        );
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
      } catch (error) {
        toast(error.message);
      } finally {
        button.disabled = false;
      }
    });
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
