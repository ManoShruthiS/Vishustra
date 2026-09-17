"""VISHUSTRA KERNEL: pipeline synthesizer.

Turns a natural-language *intention* (e.g. "clean this, make it polite, in
Hindi") into an ordered, executable pipeline of skills, plus a human-readable
explanation of every decision.

Two modes:
  - rule-mode: transparent, deterministic keyword matching over SKILLS.
  - llm-mode (optional): when GEMINI_API_KEY is available and rules are
    ambiguous, a Gemini model drafts the pipeline as JSON; we still validate
    and explain it. Falls back to rule-mode on any error.
"""

import json
import os
import re
import logging
from typing import Any, Dict, List, Tuple

from vishustra_core.kernel.skills import CATEGORY_ORDER, LANGUAGE_ALIASES, SKILLS

logger = logging.getLogger(__name__)

LANGUAGE_NAMES = list(LANGUAGE_ALIASES.keys())
TONE_NAMES = ["polite", "formal", "friendly", "professional", "sarcastic", "informal", "respectful", "kind", "soft"]


def extract_params(intent: str) -> Dict[str, Any]:
    """Pulls concrete parameters out of the free-text intention."""
    params: Dict[str, Any] = {}
    low = intent.lower()

    for lang in LANGUAGE_NAMES:
        if re.search(rf"\b({lang}|{LANGUAGE_ALIASES[lang]})\b", low):
            params["target_language"] = LANGUAGE_ALIASES[lang]
            break

    for tone in TONE_NAMES:
        if re.search(rf"\b{tone}\b", low):
            params["target_tone"] = tone
            break

    m = re.search(r"\b(\d{1,2})\s*line", low) or re.search(r"\b(\d{1,2})\s*(?:sentence|point)", low)
    if m:
        n = int(m.group(1))
        params["min_sentences"] = max(1, n)
        params["max_sentences"] = max(1, n)
        params["summary_ratio"] = 0.3

    if "json" in low:
        params["json_indent"] = 2

    # ordering: if the user wants something cleaned THEN summarized, and also
    # mentions summary earlier, this flag lets us keep sanitize-first order.
    return params


def _score_skills(intent: str) -> List[Tuple[str, str]]:
    """Returns [(skill_id, reason)] for every skill that matches the intent."""
    low = intent.lower()
    hits: List[Tuple[str, str]] = []
    for skill_id, meta in SKILLS.items():
        matched_kw = next((kw for kw in meta["keywords"] if kw in low), None)
        if matched_kw:
            hits.append((skill_id, f"matched keyword '{matched_kw}'"))
    return hits


def synthesize_rules(intent: str, text: str = "") -> Dict[str, Any]:
    """Deterministic, explainable synthesis."""
    params = extract_params(intent)
    hits = _score_skills(intent)

    # A pure "analyze only" request never wants sanitize/transform prepended,
    # but combined requests ("clean and summarize") do. Rule:
    # sanitize and transform skills are inserted in canonical category order.
    ordered = sorted(hits, key=lambda kv: CATEGORY_ORDER[SKILLS[kv[0]]["category"]])

    pipeline = [skill_id for skill_id, _ in ordered]
    # dedupe preserving order
    seen = set()
    deduped = []
    for s in pipeline:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    pipeline = deduped

    reasons = [
        f"{step}: {SKILLS[step]['description']} [{reason}]"
        for step, reason in [(s, next(r for sid, r in ordered if sid == s)) for s in pipeline]
    ]

    mode = "rules"
    # Only consult the LLM when the rules found nothing or we can't resolve a
    # language/tone that was clearly requested.
    if not pipeline:
        llm = _synthesize_llm(intent)
        if llm is not None:
            pipeline, reasons, params = llm
            mode = "llm"

    return {
        "mode": mode,
        "pipeline": pipeline,
        "reasons": reasons,
        "params": params,
        "skills": [SKILLS[s] for s in pipeline],
    }


def _synthesize_llm(intent: str):
    """Optional Gemini-backed synthesis. Returns (pipeline, reasons, params) or None."""
    if not os.environ.get("GEMINI_API_KEY"):
        return None
    try:
        from google import genai

        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    except Exception as e:  # pragma: no cover - env/import issues
        logger.debug(f"LLM synthesizer unavailable: {e}")
        return None

    skill_ids = list(SKILLS.keys())
    prompt = (
        "You are the VISHUSTRA pipeline synthesizer. Choose an ordered pipeline of skills "
        "that best fulfills the user's intention. Skills available: " + ", ".join(skill_ids) + ". "
        "Order rules: sanitize before transform before analyze before format. "
        'Reply ONLY with valid JSON: {"pipeline": ["skill1", ...], "params": {}, "reasons": ["why skill1", ...]}. '
        f'Intention: "{intent}"'
    )
    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        text = response.text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text).rstrip("`").strip()
        data = json.loads(text)
        pipeline = [s for s in data.get("pipeline", []) if s in SKILLS]
        if not pipeline:
            return None
        reasons = data.get("reasons", [f"recommended by LLM for '{skill_id}'" for skill_id in pipeline])
        params = data.get("params", {})
        if len(reasons) < len(pipeline):
            reasons += [f"recommended by LLM for '{skill_id}'" for skill_id in pipeline[len(reasons):]]
        return pipeline, reasons, params
    except Exception as e:
        logger.warning(f"LLM synthesis failed, falling back to rules: {e}")
        return None


def synthesize(intent: str, text: str = "") -> Dict[str, Any]:
    """Public entry point: intention -> pipeline plan."""
    return synthesize_rules(intent, text)