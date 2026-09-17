# VISHUSTRA Kernel REST API

Zero-dependency HTTP server (stdlib `http.server`). Default port `8812`.

```
python -m vishustra_core.main --serve [--port 8812]
```

## Endpoints

### `GET /health`

```json
{ "status": "online", "health": "OK", "service": "vishustra-kernel" }
```

### `POST /kernel`

Synthesizes and executes a pipeline for the given intention.

**Request body** (JSON):

| Field   | Type   | Required | Description                                      |
|---------|--------|----------|--------------------------------------------------|
| `intent`| string | yes      | Plain-language instruction ("clean, make polite") |
| `text`  | string | no       | Input text to process (default `""`)              |

**Response `200`:**

```json
{
  "intent": "detect sentiment and summarize",
  "mode": "rules",
  "pipeline": ["text_summarizer", "sentiment_analyzer"],
  "reasons": [
    "text_summarizer: Summarizes long text into a short extractive summary. [matched keyword 'summarize']",
    "sentiment_analyzer: Detects the sentiment (positive/negative/neutral) and a score. [matched keyword 'sentiment']"
  ],
  "params": {},
  "result": {
    "final": { "sentiment": "positive", "score": 0.9 },
    "outputs": { "text_summarizer": "..." },
    "cache": {},
    "trace": [
      { "step": "text_summarizer", "input_type": "str", "output_type": "str", "status": "ok" },
      { "step": "sentiment_analyzer", "input_type": "str", "output_type": "dict", "status": "ok" }
    ]
  }
}
```

**Error responses:**

| Status | Condition                              | Body                                   |
|--------|----------------------------------------|----------------------------------------|
| `400`  | Missing/empty `intent`                 | `{"error": "missing 'intent'"}`        |
| `422`  | Intent synthesized nothing usable      | `{"error": "Empty pipeline: ..."}`     |
| `500`  | Unexpected execution error             | `{"error": "<message>"}`               |
| `404`  | Unknown path                           | `{"error": "not found", "hint": "..."}`|

## Request/response semantics

- Output types are heterogeneous by design: `final` may be a `str`, `list`,
  `dict`, or `None` depending on the last pipeline step.
- `trace` entries record each step's input/output Python type plus `status`
  (`ok` or `error`).
- Running the server adds no additional runtime dependency beyond the stdlib.

## Examples

```bash
curl -X POST http://127.0.0.1:8812/kernel -H "Content-Type: application/json" \
  -d '{"intent":"translate to hindi and make it formal","text":"Please tell the manager about the delay."}'

curl -X POST http://127.0.0.1:8812/kernel -H "Content-Type: application/json" \
  -d '{"intent":"extract the phone numbers"}'
```