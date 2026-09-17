"""VISHUSTRA KERNEL: pipeline executor.

Executes a synthesized pipeline over input data. Handles the fact that nodes
have heterogeneous contracts (str->str, str->dict, dict->dict, str->list, ...)
by coercing data between steps, resolving per-skill context parameters, and
recording an explainable trace for every step.
"""

import importlib
import json
import logging
import os
import re
from typing import Any, Dict, List

from vishustra_core.kernel.skills import SKILLS

logger = logging.getLogger(__name__)


class VishustraKernelError(Exception):
    """Raised when a pipeline cannot be built or executed."""


def _to_text(data: Any) -> str:
    """Coerce any skill output into the most useful plain-text form."""
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        if "text" in data and isinstance(data["text"], str):
            return data["text"]
        parts = [f"{k}: {v}" for k, v in data.items() if not isinstance(v, (dict, list))]
        return "; ".join(parts) if parts else str(data)
    if isinstance(data, (list, tuple)):
        return ", ".join(str(x) for x in data)
    return str(data)


def _save_skills_registry(skill_id: str, meta: Dict[str, Any]) -> None:
    """Persist a runtime-registered skill so it survives restarts."""
    import vishustra_core.kernel.skills as skills_module

    path = skills_module._RUNTIME_SKILLS_PATH
    registry: Dict[str, Any] = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                registry = json.load(f)
        except (OSError, ValueError):
            registry = {}
    registry[skill_id] = meta
    with open(path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


def _wrap_statement(data: Any) -> Dict[str, Any]:
    """Pack free text into the {'statement': ...} shape some nodes require."""
    if isinstance(data, dict):
        return data
    return {"statement": _to_text(data)}


def _coerce_for(skill: Dict[str, Any], data: Any) -> Any:
    """Shape `data` into what the target skill accepts."""
    accepts = skill.get("accepts", "str")
    if accepts == "dict_statement":
        return _wrap_statement(data)
    if accepts == "str" and not isinstance(data, str):
        return _to_text(data)
    return data


def load_node(module_name: str, class_name: str):
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls()


class PipelineExecutor:
    """Loads skills, coerces data, runs the chain, records an explainable trace."""

    def __init__(self, plan: Dict[str, Any]) -> None:
        self.plan = plan
        self.pipeline: List[str] = plan.get("pipeline", [])
        if not self.pipeline:
            raise VishustraKernelError("Empty pipeline: nothing was synthesized from the intention.")
        self.nodes: List[Any] = []
        for skill_id in self.pipeline:
            meta = SKILLS.get(skill_id)
            if meta is None:
                raise VishustraKernelError(f"Unknown skill in pipeline: {skill_id}")
            self.nodes.append(load_node(meta["module"], meta["class"]))
        self.trace: List[Dict[str, Any]] = []

    def run(self, data: Any, user_params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        context: Dict[str, Any] = {
            "cache_store": {},
            "outputs": {},
        }
        params = dict(user_params or {})
        ctx_defaults = self.plan.get("params", {})
        params.update({k: v for k, v in ctx_defaults.items() if k not in params})

        current = data
        # `stream` is the canonical *text lane*: the most recent sanitized /
        # transformed / formatted string. Analyze steps produce dicts/lists and
        # should not clobber the text lane, otherwise downstream analyze steps
        # (e.g. keyword extraction after sentiment) would analyze metadata
        # instead of the actual text.
        stream = data if isinstance(data, str) else _to_text(data)
        context["user_params"] = dict(params)

        for step_idx, skill_id in enumerate(self.pipeline):
            meta = SKILLS[skill_id]
            node = self.nodes[step_idx]

            step_params = dict(params)
            step_params.update({k: v for k, v in meta.get("params", {}).items() if k not in step_params})
            step_context = {**context, **step_params}
            step_context["step_index"] = step_idx

            accepts = meta.get("accepts", "str")
            if accepts in ("str", "dict_statement"):
                # feed the text lane when the current value isn't plain text yet
                if isinstance(current, str):
                    source = current
                elif isinstance(stream, str):
                    source = stream
                else:
                    source = current
            else:
                source = current
            prepared = _coerce_for(meta, source)
            try:
                result = node.process(prepared, step_context)
            except Exception as e:
                logger.error(f"Step {skill_id} failed: {e}")
                self.trace.append({
                    "step": skill_id,
                    "input_type": type(prepared).__name__,
                    "status": "error",
                    "error": str(e),
                })
                raise VishustraKernelError(f"Pipeline aborted at step '{skill_id}': {e}") from e

            if meta["category"] in ("sanitize", "transform", "format") and isinstance(result, str):
                stream = result

            self.trace.append({
                "step": skill_id,
                "input_type": type(prepared).__name__,
                "output_type": type(result).__name__,
                "status": "ok",
            })
            context["outputs"][skill_id] = result
            current = result

        return {
            "final": current,
            "outputs": context["outputs"],
            "cache": context["cache_store"],
            "trace": self.trace,
        }


class VishustraKernel:
    """Facade: intention + text -> explainable pipeline execution."""

    def __init__(self) -> None:
        self.synthesizer = None

    def process(self, intent: str, data: Any, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        from vishustra_core.kernel.synthesizer import synthesize

        plan = synthesize(intent, _to_text(data) if not isinstance(data, str) else data)
        executor = PipelineExecutor(plan)
        result = executor.run(data, params)
        return {
            "intent": intent,
            "mode": plan["mode"],
            "pipeline": plan["pipeline"],
            "reasons": plan["reasons"],
            "params": plan["params"],
            "result": result,
        }

    def register_node(self, file_path: str, description: str | None = None,
                      keywords: List[str] | None = None) -> str:
        """Teach the Kernel a newly written node (used by the ayan worker).

        Imports the module from `file_path`, discovers the single BaseNode
        subclass inside it, and adds it to the live skill registry and to the
        on-disk SKILLS table so future synthesis can route to it.
        """
        import importlib.util
        import inspect
        import sys
        from vishustra_core.nodes.base_node import BaseNode

        path = os.path.abspath(file_path)
        module_name = "vishustra_core.nodes." + os.path.splitext(os.path.basename(path))[0]
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        cls = None
        for obj in vars(module).values():
            if (inspect.isclass(obj) and issubclass(obj, BaseNode)
                    and obj is not BaseNode and obj.__module__ == module.__name__):
                cls = obj
                break
        if cls is None:
            raise VishustraKernelError(f"No BaseNode subclass found in {path}")

        node = cls()
        skill_id = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", node.node_name)
        skill_id = skill_id.replace(" ", "_").replace("-", "_").lower()
        node_desc = (description
                     or (getattr(node, "__doc__", None) or "").strip().splitlines()[:1]
                     or [f"Node co-written by the ayan worker: {node.node_name}."])[0]
        node_keywords = keywords or _derive_keywords(node, cls, skill_id, node_desc) or [skill_id]

        from vishustra_core.kernel.skills import SKILLS
        SKILLS[skill_id] = {
            "id": skill_id,
            "node_name": node.node_name,
            "class": cls.__name__,
            "module": module.__name__,
            "description": node_desc,
            "keywords": node_keywords,
            "accepts": "str",
            "category": "transform",
            "runtime": True,
        }
        _save_skills_registry(skill_id, SKILLS[skill_id])
        return skill_id


_KNOWN_STOPWORDS = frozenset({
    "this", "that", "these", "those", "with", "from", "into", "about", "text",
    "data", "node", "input", "output", "the", "and", "for", "will", "were",
    "been", "are", "was", "you", "your", "our", "them", "they", "make", "use",
    "apply", "want", "need", "please", "process", "processing", "generator",
    "generate", "converter", "convert", "extractor", "extract", "formatter",
    "formatted", "validate", "validator", "classifier", "classify", "name",
})


def _derive_keywords(node, cls, skill_id: str, description: str) -> List[str]:
    """Suggest route keywords from the node_name, class name, and docstring.

    Words are lower-cased, split on non-alpha boundaries, and filtered against
    a small stopword set so generated skills route on meaningful tokens.
    """
    sources = [
        node.node_name.replace("_", " ").replace("-", " "),
        re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", cls.__name__),
        skill_id.replace("_", " "),
        description,
    ]
    words: List[str] = []
    for source in sources:
        for token in re.findall(r"[a-z]+", str(source).lower()):
            token = token.rstrip("s")
            if len(token) > 3 and token not in _KNOWN_STOPWORDS and token not in words:
                words.append(token)
    return words[:12]