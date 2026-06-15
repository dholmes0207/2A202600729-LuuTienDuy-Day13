"""Mitigation + observability wrapper for the opaque e-commerce agent."""
from __future__ import annotations

# Add _lib/ (stdlib + openai deps) to sys.path for the PyInstaller binary
import sys as _sys, os as _os
_lib = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), '_lib')
if _os.path.isdir(_lib) and _lib not in _sys.path:
    _sys.path.insert(0, _lib)


import os
import re
import time

try:
    from telemetry.logger import logger, new_correlation_id, set_correlation_id
    from telemetry.cost import cost_from_usage
    from telemetry.redact import redact
except Exception:
    logger = None
    def cost_from_usage(m, u): return 0.0
    def redact(s): return (s, 0)
    def new_correlation_id(): return "n/a"
    def set_correlation_id(c): pass

# --- Load system prompt from prompt.txt ---
_PROMPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt.txt")
try:
    with open(_PROMPT_PATH, encoding="utf-8") as _f:
        _SYSTEM_PROMPT = _f.read().strip()
except Exception:
    _SYSTEM_PROMPT = None

# --- PII patterns for output redaction ---
_PII_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PII_PHONE = re.compile(r"\b(?:\+84|0)\d{9}\b")

# --- Injection sanitizer patterns ---
_INJECT_PATS = [
    re.compile(r"GHI\s*CH[UÚ][:\s].*", re.IGNORECASE | re.DOTALL),
    re.compile(r"(?:NOTE|NOTES)[:\s].*", re.IGNORECASE | re.DOTALL),
]


def _sanitize(question):
    """Strip suspicious embedded instructions from order notes."""
    q = question
    for pat in _INJECT_PATS:
        q = pat.sub("", q)
    return q.strip() or question


def _redact_answer(answer):
    """Remove PII from the agent answer."""
    if not answer:
        return answer
    answer = _PII_EMAIL.sub("[EMAIL]", answer)
    answer = _PII_PHONE.sub("[PHONE]", answer)
    return answer


def mitigate(call_next, question, config, context):
    cid = new_correlation_id()
    set_correlation_id(cid)

    # 1. Override system prompt
    conf = dict(config)
    if _SYSTEM_PROMPT:
        conf["system_prompt"] = _SYSTEM_PROMPT

    # 2. Sanitize input (injection defense)
    clean_q = _sanitize(question)

    # 3. Call the agent with retry on failure
    result = None
    wall_ms = 0
    max_retries = 3
    for attempt in range(max_retries):
        t0 = time.time()
        try:
            result = call_next(clean_q, conf)
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(0.3 * (attempt + 1))
                continue
            return {"answer": None, "status": "wrapper_error",
                    "steps": 0, "trace": [], "meta": {}}
        wall_ms = int((time.time() - t0) * 1000)

        status = result.get("status", "ok")
        if status == "ok":
            break
        if status in ("loop", "max_steps") and attempt < max_retries - 1:
            time.sleep(0.3)
            continue
        break

    if result is None:
        return {"answer": None, "status": "wrapper_error",
                "steps": 0, "trace": [], "meta": {}}

    # 4. Redact PII from answer
    if result.get("answer"):
        result["answer"] = _redact_answer(result["answer"])

    # 5. Observability logging
    meta = result.get("meta", {})
    usage = meta.get("usage", {})
    if logger:
        logger.log_event("AGENT_CALL", {
            "qid": context.get("qid"),
            "session_id": context.get("session_id"),
            "turn_index": context.get("turn_index"),
            "status": result.get("status"),
            "latency_ms": meta.get("latency_ms"),
            "wall_ms": wall_ms,
            "tokens": usage,
            "cost_usd": cost_from_usage(meta.get("model", ""), usage),
            "tools_used": meta.get("tools_used", []),
            "tool_count": len(meta.get("tools_used", [])),
            "steps": result.get("steps"),
            "pii_in_answer": redact(result.get("answer") or "")[1] > 0,
        })

    return result
