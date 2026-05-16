package com.beyondrgb;

import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.Executors;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

public final class AppServer {
    private static final Path STATIC_DIR = Path.of("src", "main", "resources", "static");
    private static final int PORT = intEnv("UI_INTERNAL_PORT", 8501);
    private static final List<String> PROVIDERS = List.of("huggingface");

    private final Object stateLock = new Object();
    private final RuntimeState state;

    private AppServer() throws IOException, InterruptedException {
        this.state = new RuntimeState(loadDefaults());
    }

    public static void main(String[] args) throws Exception {
        AppServer app = new AppServer();
        HttpServer server = HttpServer.create(new InetSocketAddress("0.0.0.0", PORT), 0);
        server.createContext("/", app::route);
        server.setExecutor(Executors.newCachedThreadPool());
        server.start();
        System.out.println("Java UI server running on http://0.0.0.0:" + PORT);
    }

    private void route(HttpExchange exchange) throws IOException {
        try {
            String path = cleanPath(exchange.getRequestURI().getPath());
            String method = exchange.getRequestMethod();
            boolean readPage = "GET".equals(method) || "HEAD".equals(method);

            if (readPage && "/".equals(path)) {
                sendFile(exchange, STATIC_DIR.resolve("index.html"));
            } else if (readPage && "/devel".equals(path)) {
                sendFile(exchange, STATIC_DIR.resolve("devel.html"));
            } else if (readPage && path.startsWith("/static/")) {
                sendStatic(exchange, path.substring("/static/".length()));
            } else if (readPage && "/health".equals(path)) {
                sendText(exchange, 200, "ok", "text/plain; charset=utf-8");
            } else if ("GET".equals(method) && "/api/state".equals(path)) {
                sendJson(exchange, 200, statePayload());
            } else if ("GET".equals(method) && "/api/datasets".equals(path)) {
                sendJson(exchange, 200, Map.of("datasets", listDatasets()));
            } else if (readPage && "/api/data".equals(path)) {
                sendDataFile(exchange);
            } else if ("GET".equals(method) && "/api/download".equals(path)) {
                downloadReport(exchange);
            } else if ("POST".equals(method) && "/api/agent".equals(path)) {
                runAgent(exchange);
            } else if ("POST".equals(method) && "/api/settings".equals(path)) {
                saveSettings(exchange);
            } else if ("POST".equals(method) && "/api/tools/refresh".equals(path)) {
                refreshTools(exchange);
            } else if ("POST".equals(method) && "/api/models/refresh".equals(path)) {
                refreshModels(exchange);
            } else if ("POST".equals(method) && "/api/system".equals(path)) {
                systemAction(exchange);
            } else {
                sendJson(exchange, 404, Map.of("error", "Not found"));
            }
        } catch (BridgeException exc) {
            sendJson(exchange, 502, Map.of("error", exc.getMessage()));
        } catch (IllegalArgumentException exc) {
            sendJson(exchange, 400, Map.of("error", exc.getMessage()));
        } catch (Exception exc) {
            sendJson(exchange, 500, Map.of("error", exc.getClass().getSimpleName() + ": " + exc.getMessage()));
        } finally {
            exchange.close();
        }
    }

    private void runAgent(HttpExchange exchange) throws IOException, InterruptedException {
        Map<String, Object> request = parseRequestObject(exchange);
        String prompt = stringValue(request.get("prompt")).trim();
        if (prompt.isEmpty()) {
            throw new IllegalArgumentException("Enter a request before sending it.");
        }
        List<Object> selectedDatasets = listValue(request.get("datasets"));
        String agentPrompt = promptWithDatasets(prompt, selectedDatasets);

        synchronized (stateLock) {
            state.chatHistory.add(Map.of("role", "user", "content", agentPrompt));
            Map<String, Object> bridgeRequest = new LinkedHashMap<>();
            bridgeRequest.put("prompt", agentPrompt);
            bridgeRequest.put("history", state.chatHistory.subList(0, state.chatHistory.size() - 1));
            bridgeRequest.put("provider", state.provider);
            bridgeRequest.put("model", state.model);
            bridgeRequest.put("max_steps", state.maxSteps);
            bridgeRequest.put("system_prompt", state.systemPrompt);

            try {
                Map<String, Object> result = bridgeObject("run-agent", bridgeRequest, Duration.ZERO);
                Map<String, Object> assistant = objectValue(result.get("assistant_message"));
                Map<String, Object> latestResult = objectValue(result.get("latest_result"));
                Object latestTelemetry = result.get("latest_telemetry");

                state.chatHistory.add(Map.of(
                    "role", "assistant",
                    "content", stringValue(assistant.get("content"))
                ));
                state.traceHistory.add(listValue(latestResult.get("trace")));
                state.telemetryHistory.add(latestTelemetry);
                state.latestResult = latestResult;
            } catch (BridgeException exc) {
                Map<String, Object> latestResult = new LinkedHashMap<>();
                String error = "Agent error: `" + exc.getMessage() + "`";
                latestResult.put("content", error);
                latestResult.put("trace", List.of());
                latestResult.put("telemetry", null);
                latestResult.put("is_error", true);

                state.chatHistory.add(Map.of("role", "assistant", "content", error));
                state.telemetryHistory.add(null);
                state.latestResult = latestResult;
            }
        }

        sendJson(exchange, 200, statePayload());
    }

    private String promptWithDatasets(String prompt, List<Object> datasets) {
        List<String> cleanDatasets = datasets.stream()
            .map(AppServer::stringValue)
            .map(String::trim)
            .filter(value -> !value.isEmpty())
            .toList();
        if (cleanDatasets.isEmpty()) {
            return prompt;
        }

        return prompt
            + "\n\nSelected input datasets:\n"
            + String.join("\n", cleanDatasets.stream().map(value -> "- " + value).toList())
            + "\n\nUse only these selected datasets unless the user explicitly asks for other files.";
    }

    private void saveSettings(HttpExchange exchange) throws IOException {
        Map<String, Object> request = parseRequestObject(exchange);
        String provider = stringOrDefault(request.get("provider"), state.provider).trim().toLowerCase();
        if (!PROVIDERS.contains(provider)) {
            throw new IllegalArgumentException("Unsupported provider: " + provider);
        }

        String model = stringOrDefault(request.get("model"), stringValue(state.defaults.get("model"))).trim();
        int maxSteps = intValue(request.get("max_steps"), intValue(state.defaults.get("max_steps"), 6));
        String systemPrompt = stringOrDefault(request.get("system_prompt"), stringValue(state.defaults.get("system_prompt")));

        if (maxSteps < 1 || maxSteps > 20) {
            throw new IllegalArgumentException("Max steps must be between 1 and 20.");
        }

        synchronized (stateLock) {
            state.provider = provider;
            state.model = model;
            state.maxSteps = maxSteps;
            state.systemPrompt = systemPrompt;
        }
        sendJson(exchange, 200, statePayload());
    }

    private void refreshTools(HttpExchange exchange) throws IOException, InterruptedException {
        List<Object> tools = bridgeList("tools", Map.of(), Duration.ZERO);
        synchronized (stateLock) {
            state.mcpTools = tools;
        }
        sendJson(exchange, 200, statePayload());
    }

    private void refreshModels(HttpExchange exchange) throws IOException, InterruptedException {
        List<Object> models = bridgeList("models", Map.of("provider", state.provider), Duration.ZERO);
        synchronized (stateLock) {
            state.providerModels = models;
        }
        sendJson(exchange, 200, statePayload());
    }

    private void systemAction(HttpExchange exchange) throws IOException {
        Map<String, Object> request = parseRequestObject(exchange);
        String action = stringValue(request.get("action"));

        synchronized (stateLock) {
            switch (action) {
                case "clear_conversation" -> {
                    state.chatHistory.clear();
                    state.traceHistory.clear();
                    state.latestResult = null;
                }
                case "clear_telemetry" -> {
                    state.telemetryHistory.clear();
                    if (state.latestResult != null) {
                        state.latestResult.put("telemetry", null);
                    }
                }
                case "clear_tool_cache" -> {
                    state.mcpTools.clear();
                    state.providerModels.clear();
                }
                case "reset_settings" -> state.resetSettings();
                default -> throw new IllegalArgumentException("Unknown system action: " + action);
            }
        }

        sendJson(exchange, 200, statePayload());
    }

    private List<Map<String, Object>> listDatasets() throws IOException {
        Path dataRoot = dataRootPath();
        if (!Files.isDirectory(dataRoot)) {
            Path localData = Path.of("data").toAbsolutePath().normalize();
            dataRoot = Files.isDirectory(localData) ? localData : dataRoot;
        }
        if (!Files.isDirectory(dataRoot)) {
            return List.of();
        }

        Path root = dataRoot;
        List<Map<String, Object>> datasets = new ArrayList<>();
        try (var paths = Files.walk(root, 2)) {
            paths
                .filter(Files::isRegularFile)
                .filter(path -> isDatasetFile(path.getFileName().toString()))
                .sorted()
                .forEach(path -> {
                    Map<String, Object> item = new LinkedHashMap<>();
                    item.put("name", path.getFileName().toString());
                    item.put("relative_path", root.relativize(path).toString());
                    item.put("type", datasetType(path.getFileName().toString()));
                    datasets.add(item);
                });
        }
        return datasets;
    }

    private static boolean isDatasetFile(String name) {
        String lower = name.toLowerCase();
        return lower.endsWith(".zip") || lower.endsWith(".tif") || lower.endsWith(".tiff") || lower.endsWith(".csv");
    }

    private static String datasetType(String name) {
        String lower = name.toLowerCase();
        if (lower.endsWith(".zip")) return "zip";
        if (lower.endsWith(".tif") || lower.endsWith(".tiff")) return "raster";
        if (lower.endsWith(".csv")) return "table";
        return "file";
    }

    private Path dataRootPath() {
        return Path.of(stringValue(state.defaults.get("data_root"))).toAbsolutePath().normalize();
    }

    private void sendDataFile(HttpExchange exchange) throws IOException {
        String relativePath = queryParam(exchange, "path", "").replace('\\', '/');
        if (relativePath.isBlank()) {
            throw new IllegalArgumentException("Missing data file path.");
        }

        Path dataRoot = dataRootPath();
        if (!Files.isDirectory(dataRoot)) {
            Path localData = Path.of("data").toAbsolutePath().normalize();
            dataRoot = Files.isDirectory(localData) ? localData : dataRoot;
        }
        if (!Files.isDirectory(dataRoot)) {
            throw new IllegalArgumentException("Data root does not exist.");
        }

        Path root = dataRoot.toAbsolutePath().normalize();
        Path target = root.resolve(relativePath).normalize();
        if (!target.startsWith(root)) {
            throw new IllegalArgumentException("Data file path escapes the data root.");
        }
        sendFile(exchange, target);
    }

    private void downloadReport(HttpExchange exchange) throws IOException {
        String format = queryParam(exchange, "format", "md").toLowerCase();
        String content;
        synchronized (stateLock) {
            if (state.latestResult == null || Boolean.TRUE.equals(state.latestResult.get("is_error"))) {
                throw new IllegalArgumentException("No downloadable report is available.");
            }
            content = exportContent(state.latestResult);
        }

        switch (format) {
            case "md" -> sendDownload(
                exchange,
                "real-estate-beyond-rgb-report.md",
                "text/markdown; charset=utf-8",
                content.getBytes(StandardCharsets.UTF_8)
            );
            case "pdf" -> sendDownload(
                exchange,
                "real-estate-beyond-rgb-report.pdf",
                "application/pdf",
                pdfBytes(content)
            );
            case "docx" -> sendDownload(
                exchange,
                "real-estate-beyond-rgb-report.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                docxBytes(content)
            );
            default -> throw new IllegalArgumentException("Unsupported download format: " + format);
        }
    }

    private String exportContent(Map<String, Object> latestResult) {
        StringBuilder content = new StringBuilder();
        content.append("# Real Estate Beyond RGB Report\n\n");
        content.append(stringValue(latestResult.get("content")).strip()).append("\n");

        List<Object> trace = listValue(latestResult.get("trace"));
        if (!trace.isEmpty()) {
            content.append("\n\n## Data Used\n");
            int index = 1;
            for (Object item : trace) {
                Map<String, Object> step = objectValue(item);
                content.append("\n### ")
                    .append(index++)
                    .append(". ")
                    .append(stringValue(step.get("tool")))
                    .append("\n\n");
                content.append("Arguments:\n\n```json\n")
                    .append(Json.stringify(step.getOrDefault("arguments", Map.of())))
                    .append("\n```\n\n");
                if (step.get("result_preview") != null) {
                    content.append("Result data:\n\n```text\n")
                        .append(stringValue(step.get("result_preview")))
                        .append("\n```\n");
                }
            }
        }

        return content.toString();
    }

    private Map<String, Object> statePayload() {
        synchronized (stateLock) {
            Map<String, Object> payload = new LinkedHashMap<>();
            payload.put("runtime", runtimePayload());
            payload.put("counts", Map.of(
                "messages", state.chatHistory.size(),
                "telemetry", state.telemetryHistory.size(),
                "tools", state.mcpTools.size(),
                "models", state.providerModels.size()
            ));

            Map<String, Object> latestResult = state.latestResult == null ? null : new LinkedHashMap<>(state.latestResult);
            if (latestResult != null) {
                latestResult.put("telemetry", compactTelemetry(latestResult.get("telemetry")));
            }

            payload.put("latest_result", latestResult);
            payload.put("latest_telemetry", latestTelemetry());
            payload.put("mcp_tools", state.mcpTools);
            payload.put("provider_models", state.providerModels);
            return payload;
        }
    }

    private Map<String, Object> runtimePayload() {
        Map<String, Object> runtime = new LinkedHashMap<>();
        runtime.put("provider", state.provider);
        runtime.put("model", state.model);
        runtime.put("max_steps", state.maxSteps);
        runtime.put("system_prompt", state.systemPrompt);
        runtime.put("data_root", state.defaults.get("data_root"));
        runtime.put("mcp_server", state.defaults.get("mcp_server"));
        runtime.put("hf_api_base", state.defaults.get("hf_api_base"));
        runtime.put("user_route", "/");
        runtime.put("developer_route", "/devel");
        return runtime;
    }

    private Object latestTelemetry() {
        for (int i = state.telemetryHistory.size() - 1; i >= 0; i--) {
            Object telemetry = state.telemetryHistory.get(i);
            if (telemetry != null) {
                return compactTelemetry(telemetry);
            }
        }
        return null;
    }

    @SuppressWarnings("unchecked")
    private Object compactTelemetry(Object telemetryValue) {
        if (!(telemetryValue instanceof Map<?, ?> telemetry)) {
            return null;
        }
        Map<String, Object> latest = latestCall((Map<String, Object>) telemetry);
        Map<String, Object> compact = new LinkedHashMap<>();
        compact.put("provider", telemetry.get("provider"));
        compact.put("model", telemetry.get("model"));
        compact.put("status_code", latest.get("status_code"));
        compact.put("calls", listValue(telemetry.get("calls")).size());
        compact.put("usage", latest.getOrDefault("usage", Map.of()));
        compact.put("rate_limit_headers", latest.getOrDefault("rate_limit_headers", Map.of()));
        compact.put("rate_limit_error", latest.getOrDefault("rate_limit_error", Map.of()));
        compact.put("max_completion_tokens", telemetry.get("max_completion_tokens"));
        compact.put("error", latest.get("error"));
        compact.put("raw", telemetry);
        return compact;
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> latestCall(Map<String, Object> telemetry) {
        List<Object> calls = listValue(telemetry.get("calls"));
        if (!calls.isEmpty() && calls.get(calls.size() - 1) instanceof Map<?, ?> call) {
            return (Map<String, Object>) call;
        }
        return telemetry;
    }

    private static Map<String, Object> loadDefaults() throws IOException, InterruptedException {
        return bridgeObject("defaults", Map.of(), Duration.ofSeconds(20));
    }

    private static Map<String, Object> bridgeObject(String command, Map<String, Object> payload, Duration timeout)
        throws IOException, InterruptedException {
        Object data = bridge(command, payload, timeout);
        return objectValue(data);
    }

    private static List<Object> bridgeList(String command, Map<String, Object> payload, Duration timeout)
        throws IOException, InterruptedException {
        Object data = bridge(command, payload, timeout);
        return listValue(data);
    }

    @SuppressWarnings("unchecked")
    private static Object bridge(String command, Map<String, Object> payload, Duration timeout)
        throws IOException, InterruptedException {
        ProcessBuilder builder = new ProcessBuilder("python", "-m", "src.agent.bridge", command);
        builder.directory(Path.of(".").toAbsolutePath().normalize().toFile());
        builder.environment().putIfAbsent("PYTHONPATH", Path.of(".").toAbsolutePath().normalize().toString());

        Process process = builder.start();
        try (OutputStream stdin = process.getOutputStream()) {
            stdin.write(Json.stringify(payload).getBytes(StandardCharsets.UTF_8));
        }

        boolean finished;
        if (timeout == null || timeout.isZero() || timeout.isNegative()) {
            finished = true;
            process.waitFor();
        } else {
            finished = process.waitFor(timeout.toMillis(), java.util.concurrent.TimeUnit.MILLISECONDS);
        }

        if (!finished) {
            process.destroyForcibly();
            throw new BridgeException("Python bridge timed out for command: " + command);
        }

        String stdout = readAll(process.getInputStream());
        String stderr = readAll(process.getErrorStream());
        Object parsed = stdout.isBlank() ? Map.of() : Json.parse(stdout);

        if (process.exitValue() != 0) {
            String message = stderr.isBlank() ? stdout : stderr;
            Object errorPayload = null;
            try {
                errorPayload = Json.parse(message);
            } catch (RuntimeException ignored) {
                // Use raw process output below.
            }
            if (errorPayload instanceof Map<?, ?> errorMap && errorMap.get("error") != null) {
                throw new BridgeException(stringValue(errorMap.get("error")));
            }
            throw new BridgeException(message.strip());
        }

        Map<String, Object> envelope = (Map<String, Object>) parsed;
        if (!Boolean.TRUE.equals(envelope.get("ok"))) {
            throw new BridgeException(stringValue(envelope.get("error")));
        }
        return envelope.get("data");
    }

    private static Map<String, Object> parseRequestObject(HttpExchange exchange) throws IOException {
        String body = new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
        if (body.isBlank()) {
            return new LinkedHashMap<>();
        }
        return objectValue(Json.parse(body));
    }

    private static void sendStatic(HttpExchange exchange, String rawRelativePath) throws IOException {
        String relativePath = URLDecoder.decode(rawRelativePath, StandardCharsets.UTF_8);
        Path target = STATIC_DIR.resolve(relativePath).normalize();
        if (!target.startsWith(STATIC_DIR.normalize()) || !Files.isRegularFile(target)) {
            sendJson(exchange, 404, Map.of("error", "Static file not found"));
            return;
        }
        sendFile(exchange, target);
    }

    private static void sendDownload(HttpExchange exchange, String filename, String contentType, byte[] bytes) throws IOException {
        Headers headers = exchange.getResponseHeaders();
        headers.set("Content-Type", contentType);
        headers.set("Content-Disposition", "attachment; filename=\"" + filename + "\"");
        exchange.sendResponseHeaders(200, bytes.length);
        try (OutputStream output = exchange.getResponseBody()) {
            output.write(bytes);
        }
    }

    private static String queryParam(HttpExchange exchange, String name, String fallback) {
        String query = exchange.getRequestURI().getRawQuery();
        if (query == null || query.isBlank()) {
            return fallback;
        }
        for (String pair : query.split("&")) {
            String[] parts = pair.split("=", 2);
            String key = URLDecoder.decode(parts[0], StandardCharsets.UTF_8);
            if (!name.equals(key)) {
                continue;
            }
            return parts.length > 1 ? URLDecoder.decode(parts[1], StandardCharsets.UTF_8) : "";
        }
        return fallback;
    }

    private static byte[] pdfBytes(String content) {
        String text = content
            .replace("\r", "")
            .replace("\t", "    ");
        List<String> lines = wrapLines(text, 86);
        List<String> pageStreams = new ArrayList<>();
        for (int start = 0; start < lines.size(); start += 54) {
            StringBuilder page = new StringBuilder();
            page.append("BT\n/F1 10 Tf\n14 TL\n50 780 Td\n");
            for (String line : lines.subList(start, Math.min(start + 54, lines.size()))) {
                page.append(pdfText(line)).append(" Tj\nT*\n");
            }
            page.append("ET\n");
            pageStreams.add(page.toString());
        }
        if (pageStreams.isEmpty()) {
            pageStreams.add("BT\n/F1 10 Tf\n14 TL\n50 780 Td\n(Empty report) Tj\nET\n");
        }

        String header = "%PDF-1.4\n";
        List<String> objects = new ArrayList<>();
        objects.add("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

        int firstPageObject = 4;
        StringBuilder kids = new StringBuilder();
        for (int i = 0; i < pageStreams.size(); i++) {
            kids.append(firstPageObject + (i * 2)).append(" 0 R ");
        }
        objects.add("2 0 obj\n<< /Type /Pages /Kids [" + kids + "] /Count " + pageStreams.size() + " >>\nendobj\n");
        objects.add("3 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n");
        for (int i = 0; i < pageStreams.size(); i++) {
            int pageObject = firstPageObject + (i * 2);
            int contentObject = pageObject + 1;
            String streamText = pageStreams.get(i);
            int length = streamText.getBytes(StandardCharsets.UTF_8).length;
            objects.add(pageObject + " 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents " + contentObject + " 0 R >>\nendobj\n");
            objects.add(contentObject + " 0 obj\n<< /Length " + length + " >>\nstream\n" + streamText + "endstream\nendobj\n");
        }

        StringBuilder pdf = new StringBuilder(header);
        List<Integer> offsets = new ArrayList<>();
        for (String object : objects) {
            offsets.add(pdf.toString().getBytes(StandardCharsets.UTF_8).length);
            pdf.append(object);
        }
        int xrefOffset = pdf.toString().getBytes(StandardCharsets.UTF_8).length;
        pdf.append("xref\n0 ").append(objects.size() + 1).append("\n");
        pdf.append("0000000000 65535 f \n");
        for (int offset : offsets) {
            pdf.append(String.format("%010d 00000 n \n", offset));
        }
        pdf.append("trailer\n<< /Size ").append(objects.size() + 1).append(" /Root 1 0 R >>\n");
        pdf.append("startxref\n").append(xrefOffset).append("\n%%EOF\n");
        return pdf.toString().getBytes(StandardCharsets.UTF_8);
    }

    private static String pdfText(String text) {
        return "("
            + text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            + ")";
    }

    private static byte[] docxBytes(String content) throws IOException {
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        try (ZipOutputStream zip = new ZipOutputStream(output, StandardCharsets.UTF_8)) {
            zipEntry(zip, "[Content_Types].xml", """
                <?xml version="1.0" encoding="UTF-8"?>
                <Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
                  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
                  <Default Extension="xml" ContentType="application/xml"/>
                  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
                </Types>
                """);
            zipEntry(zip, "_rels/.rels", """
                <?xml version="1.0" encoding="UTF-8"?>
                <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
                  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
                </Relationships>
                """);
            zipEntry(zip, "word/document.xml", wordDocument(content));
        }
        return output.toByteArray();
    }

    private static String wordDocument(String content) {
        StringBuilder body = new StringBuilder();
        for (String line : content.replace("\r", "").split("\n")) {
            body.append("<w:p><w:r><w:t xml:space=\"preserve\">")
                .append(xmlEscape(line))
                .append("</w:t></w:r></w:p>");
        }
        return """
            <?xml version="1.0" encoding="UTF-8" standalone="yes"?>
            <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
              <w:body>
            """
            + body
            + """
                <w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/></w:sectPr>
              </w:body>
            </w:document>
            """;
    }

    private static void zipEntry(ZipOutputStream zip, String name, String content) throws IOException {
        zip.putNextEntry(new ZipEntry(name));
        zip.write(content.stripIndent().getBytes(StandardCharsets.UTF_8));
        zip.closeEntry();
    }

    private static List<String> wrapLines(String text, int width) {
        List<String> lines = new ArrayList<>();
        for (String paragraph : text.split("\n")) {
            String remaining = paragraph;
            while (remaining.length() > width) {
                int split = remaining.lastIndexOf(' ', width);
                if (split < 24) split = width;
                lines.add(remaining.substring(0, split).strip());
                remaining = remaining.substring(split).strip();
            }
            lines.add(remaining);
        }
        return lines;
    }

    private static String xmlEscape(String text) {
        return text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\"", "&quot;");
    }

    private static void sendFile(HttpExchange exchange, Path path) throws IOException {
        if (!Files.isRegularFile(path)) {
            sendJson(exchange, 404, Map.of("error", "File not found"));
            return;
        }

        byte[] bytes = Files.readAllBytes(path);
        Headers headers = exchange.getResponseHeaders();
        headers.set("Content-Type", mimeType(path));
        if ("HEAD".equals(exchange.getRequestMethod())) {
            exchange.sendResponseHeaders(200, -1);
            return;
        }
        exchange.sendResponseHeaders(200, bytes.length);
        try (OutputStream output = exchange.getResponseBody()) {
            output.write(bytes);
        }
    }

    private static void sendJson(HttpExchange exchange, int status, Object payload) throws IOException {
        sendText(exchange, status, Json.stringify(payload), "application/json; charset=utf-8");
    }

    private static void sendText(HttpExchange exchange, int status, String text, String contentType) throws IOException {
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Content-Type", contentType);
        if ("HEAD".equals(exchange.getRequestMethod())) {
            exchange.sendResponseHeaders(status, -1);
            return;
        }
        exchange.sendResponseHeaders(status, bytes.length);
        try (OutputStream output = exchange.getResponseBody()) {
            output.write(bytes);
        }
    }

    private static String mimeType(Path path) {
        String file = path.getFileName().toString();
        if (file.endsWith(".html")) return "text/html; charset=utf-8";
        if (file.endsWith(".css")) return "text/css; charset=utf-8";
        if (file.endsWith(".js")) return "application/javascript; charset=utf-8";
        if (file.endsWith(".svg")) return "image/svg+xml";
        if (file.endsWith(".png")) return "image/png";
        if (file.endsWith(".pdf")) return "application/pdf";
        if (file.endsWith(".json")) return "application/json; charset=utf-8";
        return "application/octet-stream";
    }

    private static String cleanPath(String path) {
        if (path == null || path.isBlank()) {
            return "/";
        }
        return path.endsWith("/") && path.length() > 1 ? path.substring(0, path.length() - 1) : path;
    }

    private static String readAll(InputStream input) throws IOException {
        return new String(input.readAllBytes(), StandardCharsets.UTF_8);
    }

    private static int intEnv(String name, int defaultValue) {
        String raw = System.getenv(name);
        if (raw == null || raw.isBlank()) {
            return defaultValue;
        }
        try {
            return Integer.parseInt(raw);
        } catch (NumberFormatException exc) {
            return defaultValue;
        }
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> objectValue(Object value) {
        if (value instanceof Map<?, ?> map) {
            return (Map<String, Object>) map;
        }
        return new LinkedHashMap<>();
    }

    @SuppressWarnings("unchecked")
    private static List<Object> listValue(Object value) {
        if (value instanceof List<?> list) {
            return (List<Object>) list;
        }
        return new ArrayList<>();
    }

    private static String stringValue(Object value) {
        return value == null ? "" : String.valueOf(value);
    }

    private static String stringOrDefault(Object value, String fallback) {
        String text = stringValue(value);
        return text.isEmpty() ? fallback : text;
    }

    private static int intValue(Object value, int fallback) {
        if (value instanceof Number number) {
            return number.intValue();
        }
        try {
            return Integer.parseInt(stringValue(value));
        } catch (NumberFormatException exc) {
            return fallback;
        }
    }

    private static final class RuntimeState {
        private final Map<String, Object> defaults;
        private String provider;
        private String model;
        private int maxSteps;
        private String systemPrompt;
        private List<Object> mcpTools = new ArrayList<>();
        private List<Object> providerModels = new ArrayList<>();
        private final List<Map<String, Object>> chatHistory = new ArrayList<>();
        private final List<List<Object>> traceHistory = new ArrayList<>();
        private final List<Object> telemetryHistory = new ArrayList<>();
        private Map<String, Object> latestResult;

        private RuntimeState(Map<String, Object> defaults) {
            this.defaults = defaults;
            resetSettings();
        }

        private void resetSettings() {
            provider = stringOrDefault(defaults.get("provider"), "huggingface");
            model = stringValue(defaults.get("model"));
            maxSteps = intValue(defaults.get("max_steps"), 6);
            systemPrompt = stringValue(defaults.get("system_prompt"));
        }
    }

    private static final class BridgeException extends RuntimeException {
        private BridgeException(String message) {
            super(message == null || message.isBlank() ? "Python bridge failed." : message);
        }
    }

    private static final class Json {
        private Json() {
        }

        private static Object parse(String text) {
            return new Parser(text).parse();
        }

        private static String stringify(Object value) {
            StringBuilder builder = new StringBuilder();
            write(builder, value);
            return builder.toString();
        }

        @SuppressWarnings("unchecked")
        private static void write(StringBuilder builder, Object value) {
            if (value == null) {
                builder.append("null");
            } else if (value instanceof String text) {
                writeString(builder, text);
            } else if (value instanceof Number || value instanceof Boolean) {
                builder.append(value);
            } else if (value instanceof Map<?, ?> map) {
                builder.append('{');
                boolean first = true;
                for (Map.Entry<?, ?> entry : map.entrySet()) {
                    if (!first) builder.append(',');
                    writeString(builder, String.valueOf(entry.getKey()));
                    builder.append(':');
                    write(builder, entry.getValue());
                    first = false;
                }
                builder.append('}');
            } else if (value instanceof Iterable<?> iterable) {
                builder.append('[');
                boolean first = true;
                for (Object item : iterable) {
                    if (!first) builder.append(',');
                    write(builder, item);
                    first = false;
                }
                builder.append(']');
            } else if (value.getClass().isArray()) {
                builder.append('[');
                Object[] array = (Object[]) value;
                for (int i = 0; i < array.length; i++) {
                    if (i > 0) builder.append(',');
                    write(builder, array[i]);
                }
                builder.append(']');
            } else {
                writeString(builder, String.valueOf(value));
            }
        }

        private static void writeString(StringBuilder builder, String text) {
            builder.append('"');
            for (int i = 0; i < text.length(); i++) {
                char c = text.charAt(i);
                switch (c) {
                    case '"' -> builder.append("\\\"");
                    case '\\' -> builder.append("\\\\");
                    case '\b' -> builder.append("\\b");
                    case '\f' -> builder.append("\\f");
                    case '\n' -> builder.append("\\n");
                    case '\r' -> builder.append("\\r");
                    case '\t' -> builder.append("\\t");
                    default -> {
                        if (c < 0x20) {
                            builder.append(String.format("\\u%04x", (int) c));
                        } else {
                            builder.append(c);
                        }
                    }
                }
            }
            builder.append('"');
        }

        private static final class Parser {
            private final String text;
            private int index;

            private Parser(String text) {
                this.text = Objects.requireNonNull(text);
            }

            private Object parse() {
                Object value = readValue();
                skipWhitespace();
                if (index != text.length()) {
                    throw new IllegalArgumentException("Unexpected JSON content at offset " + index);
                }
                return value;
            }

            private Object readValue() {
                skipWhitespace();
                if (index >= text.length()) {
                    throw new IllegalArgumentException("Unexpected end of JSON.");
                }
                char c = text.charAt(index);
                return switch (c) {
                    case '{' -> readObject();
                    case '[' -> readArray();
                    case '"' -> readString();
                    case 't' -> readLiteral("true", Boolean.TRUE);
                    case 'f' -> readLiteral("false", Boolean.FALSE);
                    case 'n' -> readLiteral("null", null);
                    default -> readNumber();
                };
            }

            private Map<String, Object> readObject() {
                index++;
                Map<String, Object> map = new LinkedHashMap<>();
                skipWhitespace();
                if (peek('}')) {
                    index++;
                    return map;
                }
                while (true) {
                    String key = readString();
                    skipWhitespace();
                    expect(':');
                    map.put(key, readValue());
                    skipWhitespace();
                    if (peek('}')) {
                        index++;
                        return map;
                    }
                    expect(',');
                    skipWhitespace();
                }
            }

            private List<Object> readArray() {
                index++;
                List<Object> list = new ArrayList<>();
                skipWhitespace();
                if (peek(']')) {
                    index++;
                    return list;
                }
                while (true) {
                    list.add(readValue());
                    skipWhitespace();
                    if (peek(']')) {
                        index++;
                        return list;
                    }
                    expect(',');
                }
            }

            private String readString() {
                expect('"');
                StringBuilder builder = new StringBuilder();
                while (index < text.length()) {
                    char c = text.charAt(index++);
                    if (c == '"') {
                        return builder.toString();
                    }
                    if (c != '\\') {
                        builder.append(c);
                        continue;
                    }
                    if (index >= text.length()) {
                        throw new IllegalArgumentException("Invalid JSON escape.");
                    }
                    char escaped = text.charAt(index++);
                    switch (escaped) {
                        case '"' -> builder.append('"');
                        case '\\' -> builder.append('\\');
                        case '/' -> builder.append('/');
                        case 'b' -> builder.append('\b');
                        case 'f' -> builder.append('\f');
                        case 'n' -> builder.append('\n');
                        case 'r' -> builder.append('\r');
                        case 't' -> builder.append('\t');
                        case 'u' -> {
                            String hex = text.substring(index, index + 4);
                            builder.append((char) Integer.parseInt(hex, 16));
                            index += 4;
                        }
                        default -> throw new IllegalArgumentException("Invalid JSON escape: \\" + escaped);
                    }
                }
                throw new IllegalArgumentException("Unterminated JSON string.");
            }

            private Object readNumber() {
                int start = index;
                if (peek('-')) index++;
                while (index < text.length() && Character.isDigit(text.charAt(index))) index++;
                boolean decimal = false;
                if (peek('.')) {
                    decimal = true;
                    index++;
                    while (index < text.length() && Character.isDigit(text.charAt(index))) index++;
                }
                if (index < text.length() && (text.charAt(index) == 'e' || text.charAt(index) == 'E')) {
                    decimal = true;
                    index++;
                    if (index < text.length() && (text.charAt(index) == '+' || text.charAt(index) == '-')) index++;
                    while (index < text.length() && Character.isDigit(text.charAt(index))) index++;
                }
                String number = text.substring(start, index);
                return decimal ? Double.parseDouble(number) : Long.parseLong(number);
            }

            private Object readLiteral(String literal, Object value) {
                if (!text.startsWith(literal, index)) {
                    throw new IllegalArgumentException("Invalid JSON literal at offset " + index);
                }
                index += literal.length();
                return value;
            }

            private void skipWhitespace() {
                while (index < text.length() && Character.isWhitespace(text.charAt(index))) {
                    index++;
                }
            }

            private boolean peek(char expected) {
                return index < text.length() && text.charAt(index) == expected;
            }

            private void expect(char expected) {
                if (!peek(expected)) {
                    throw new IllegalArgumentException("Expected '" + expected + "' at JSON offset " + index);
                }
                index++;
            }
        }
    }
}
