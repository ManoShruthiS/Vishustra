# VISHUSTRA

A modular text-processing framework in Python. Describe what you want done to
any text in plain language, and the **Kernel** synthesizes an ordered pipeline
of small, independently-testable **nodes**, executes it, and explains every
decision.

```
"clean this review, hide personal info, make it polite and short"
        ┌──────────────┐   pipeline    ┌──────────────┐
intent ─▶  SYNTHESIZER  ──────────────▶  EXECUTOR     ─▶ explainable report
        │  (rules /    │               │  (nodes +    │    pipeline, reasons,
        │   optional   │               │   coercion,  │    params, trace, final
        │   LLM)       │               │   text-lane) │
        └──────────────┘               └──────────────┘
```

## Quickstart

```bash
pip install -r requirements-dev.txt
pytest                      # run the test suite
python -m vishustra_core    # dashboard + Kernel status card
```

### Run a pipeline from the CLI

```bash
python -m vishustra_core.main --kernel \
    "clean this, remove private info, summarize to 3 lines, make it polite" \
    "This is SHIT service!! My phone 9840012345 emailed john@x.com. I waited 45 min for a cab that never came."
```

### Or as a REST API

```bash
python -m vishustra_core.main --serve --port 8812
curl -X POST http://127.0.0.1:8812/kernel -H "Content-Type: application/json" \
  -d '{"intent":"detect sentiment and summarize","text":"I love this product, it is amazing!"}'
```

Example response:

```json
{
  "intent": "detect sentiment and summarize",
  "mode": "rules",
  "pipeline": ["text_summarizer", "sentiment_analyzer"],
  "reasons": ["..."],
  "params": {},
  "result": {
    "final": {"sentiment": "positive", "score": 0.9},
    "outputs": {},
    "cache": {},
    "trace": [{"step": "text_summarizer", "input_type": "str", "output_type": "str", "status": "ok"}]
  }
}
```

### Built-in demo scenarios

```bash
python -m vishustra_core.kernel.demo --scenario review   # clean + PII + sentiment + keywords
python -m vishustra_core.kernel.demo --scenario polite   # tonal rewrite
python -m vishustra_core.kernel.demo --scenario hindi    # translate + formal tone
python -m vishustra_core.kernel.demo --scenario facts    # summarize + fact-check
python -m vishustra_core.kernel.demo --scenario urls     # URL extraction + summary
```

### Direct usage

```python
from vishustra_core.kernel import VishustraKernel

report = VishustraKernel().process(
    "clean this up, hide personal info, summarize to 3 lines, make it polite",
    "Your app is SHIT. email me at x@y.com. Support took 3 days to reply!!",
)
print("pipeline:", " -> ".join(report["pipeline"]))
print("final:   ", report["result"]["final"])
```

## Architecture

| Layer | Path | Responsibility |
|---|---|---|
| Nodes | `vishustra_core/nodes/` | One capability per `BaseNode` subclass; heterogeneous contracts (`str→str`, `str→dict`, `str→list`, …). |
| Skills registry | `vishustra_core/kernel/skills.py` | Every node's contract, keywords, and params — the single source of truth the synthesizer reasons over. Runtime-registered skills persist to `runtime_skills.json`. |
| Synthesizer | `vishustra_core/kernel/synthesizer.py` | Intention → ordered pipeline + reasons + params. Deterministic `rules` mode; optional LLM cascade when rules are ambiguous. |
| Executor | `vishustra_core/kernel/executor.py` | Loads nodes, coerces data between contracts, maintains a **text lane** so analyze steps read the real text, records a per-step trace. |
| Facade | `vishustra_core/kernel/__init__.py` | `VishustraKernel.process(intent, text)`; `register_node()` grows the skill library at runtime. |
| Dashboard/API | `vishustra_core/main.py` | Dashboard card, `--kernel` CLI, stdlib REST server. |
| Worker | `ayan.py` | LLM co-worker that writes new nodes, commits them, and registers them into the Kernel. |

## Key design ideas

- **Registry is the contract.** Nodes don't know about routing; all routing
  knowledge lives in `SKILLS`. The Kernel stays LLM-agnostic.
- **Heterogeneous contracts are accepted, not normalized.** The executor
  coerces *between* steps instead of forcing one shape on everyone.
- **Text lane vs analyze lane.** Once a `sanitize`/`transform` step has run,
  downstream `analyze` steps (e.g. keyword extraction) read the cleaned text —
  not the previous analyze node's `dict`.
- **Explainability is first-class.** Every pipeline carries the *why*, every
  execution carries a *trace*.

## Development

```bash
python check_nodes.py            # validate every node (syntax, import, patterns)
ruff check .                     # lint
pytest -v                        # full test suite
```

See `RESEARCH.md` for the research outline, current status, and next steps,
and `docs/api.md` for the REST contract.

## License

TBD — not yet decided.