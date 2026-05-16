# Hugging Face + MCP Data Agent

Dockerized starter system for a web-based agent that can inspect files in a host-mounted `./data` folder through MCP tools.

The system supports hosted inference through **Hugging Face Inference Providers**.

## Architecture

```text
host ./data
   |
   | mounted as /workspace/data
   v
agent-ui  <---- MCP Streamable HTTP ---->  mcp-tools
Java HTTP server + HTML/CSS/JS         data tools
Python model agent bridge
   |
   | OpenAI-compatible chat/tool API
   v
Hugging Face hosted inference
```

## Requirements

- Docker Engine with Docker Compose v2
- A Hugging Face token from `https://huggingface.co/settings/tokens`
- Internet access from the `agent-ui` container

## Quick Start

1. Create your environment file:

```bash
cp env.example .env
```

2. Edit `.env` and set your Hugging Face token.

```env
MODEL_PROVIDER=huggingface
HF_TOKEN=your_hugging_face_token_here
HF_MODEL=google/gemma-3-27b-it
```

3. Start the system:

```bash
docker compose up --build -d
```

4. Open the UI:

```text
http://localhost:8501
```

5. Put files in the host folder:

```text
./data
```

6. Try prompts such as:

```text
What files are in the shared data folder?
Read test-data-image.txt.
Search the data folder for testing.
```

## Services

- `agent-ui`: Java HTTP server serving the HTML/CSS/JavaScript UI and JSON API. It calls the Python agent bridge for model/tool execution.
- `mcp-tools`: MCP Streamable HTTP server exposing file/data tools.

Check status:

```bash
docker compose ps
```

Watch logs:

```bash
docker compose logs -f agent-ui mcp-tools
```

Stop:

```bash
docker compose down
```

## Configuration

Main `.env` settings:

```env
HOST_DATA_DIR=./data
CONTAINER_DATA_DIR=/workspace/data

MODEL_PROVIDER=huggingface
HF_API_BASE=https://router.huggingface.co/v1
HF_TOKEN=your_hugging_face_token_here
HF_MODEL=google/gemma-3-27b-it
HF_MAX_COMPLETION_TOKENS=4096

ALLOW_WRITE_TO_DATA=false
AGENT_MAX_STEPS=20

UI_PORT=8501
MCP_PORT=8000
```

Hugging Face's OpenAI-compatible router uses `https://router.huggingface.co/v1`.

## UI Usage

The public Java web application at `http://localhost:8501` lets you:

- Send an investment-analysis prompt to the agent.
- Review the condensed report returned by the model.
- Inspect the tool-call activity behind the analysis.

The developer console at `http://localhost:8501/devel` lets you:

- Change the model name, max tool-call steps, and system prompt.
- Show provider models.
- Show MCP tools as visual cards.
- Inspect telemetry for the latest model call.
- Clear conversation, telemetry, and cached tool/model data.

The prompt form sends requests to the Java API. The Java backend calls the Python agent bridge, and when the model chooses a tool, the agent calls the MCP server, receives the result, and sends that result back to the model.

## Data Folder

The host path from `HOST_DATA_DIR` is mounted into containers at `CONTAINER_DATA_DIR`.

Default:

```text
host:      ./data
container: /workspace/data
```

Bundled tools include:

- `list_data_files`
- `read_text_file`
- `write_text_file`
- `summarize_csv`
- `search_text_files`
- `get_data_root_info`
- `inspect_geotiff`
- `analyze_enmap_parcel`
- `compare_enmap_parcels`
- `create_parcel_metric_chart`
- `build_parcel_visualization`
- `plot_ph_profile`
- `plot_nitrogen_profile`
- `plot_phosphorus_profile`
- `plot_potassium_profile`
- `plot_magnesium_profile`
- `plot_som_profile`
- `plot_ndvi_vs_swi_scatter`
- `render_agromanagement_textbox`
- `create_agromanagement_plot_suite`
- `create_geo_and_risk_report`

Writes are disabled by default. Enable only when needed:

```env
ALLOW_WRITE_TO_DATA=true
```

## Add MCP Tools

Edit:

```text
src/mcp_server/server.py
```

Example:

```python
@mcp.tool()
def my_tool(input_text: str) -> str:
    """Return the input text in uppercase.

    Args:
        input_text: Text to transform.
    """
    return input_text.upper()
```

Rebuild:

```bash
docker compose up --build -d mcp-tools agent-ui
```

## Troubleshooting

### `HF_TOKEN is not set`

Add your Hugging Face token to `.env`, then restart:

```bash
docker compose up -d agent-ui
```

### Hugging Face model/provider error

The Hugging Face router may reject a model if no provider is available for your account or if the selected provider does not support the requested chat/tool features. Try listing models from the UI or switch to another model available to your account.

### MCP tools do not list

Check the MCP server:

```bash
docker compose logs --tail=100 mcp-tools
docker compose ps mcp-tools
```

Verify from inside the UI container:

```bash
docker compose exec agent-ui python -c "import asyncio; from src.agent.agent import list_mcp_tools; print(asyncio.run(list_mcp_tools()))"
```

### UI does not open

Check the port in `.env`:

```env
UI_PORT=8501
```

Then inspect logs:

```bash
docker compose logs --tail=100 agent-ui
```

### Data files are missing in the agent

Confirm files exist on the host:

```bash
ls -la data
```

Confirm the mount inside the container:

```bash
docker compose exec agent-ui ls -la /workspace/data
docker compose exec mcp-tools ls -la /workspace/data
```

## Development

Run syntax checks:

```bash
python -m py_compile src/agent/agent.py src/agent/bridge.py src/mcp_server/server.py src/common_config.py
javac -d build/classes $(find src/main/java -name "*.java")
```

Rebuild after code changes:

```bash
docker compose up --build -d
```

Shell into the UI container:

```bash
docker compose exec agent-ui bash
```

## Notes

- Do not commit any `.env` with real API keys. use the `.env.example` as reference.
- Keep `ALLOW_WRITE_TO_DATA=false` unless write access is required.
- Do not expose the MCP server publicly without authentication.
- Add authentication or a reverse proxy before using the UI outside localhost.
