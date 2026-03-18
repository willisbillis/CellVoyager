"""Helpers for querying and managing a local Ollama server."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Optional


DEFAULT_BASE_URL = "http://localhost:11434"


def list_local_models(base_url: str = DEFAULT_BASE_URL, timeout: float = 3.0) -> list[str]:
    """Return sorted list of model names currently downloaded in Ollama.

    Returns an empty list if Ollama is unreachable.
    """
    try:
        req = urllib.request.Request(f"{base_url}/api/tags")
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — URL is user-configured local server
            data = json.loads(resp.read())
        names = sorted({m["name"] for m in data.get("models", [])})
        return names
    except Exception:
        return []


def is_ollama_reachable(base_url: str = DEFAULT_BASE_URL, timeout: float = 2.0) -> bool:
    """Quick health-check: can we talk to the Ollama server?"""
    try:
        req = urllib.request.Request(f"{base_url}/api/tags")
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return resp.status == 200
    except Exception:
        return False


def resolve_model_name(bare_name: str, base_url: str = DEFAULT_BASE_URL, timeout: float = 3.0) -> str:
    """Resolve a bare Ollama model name to its full name with tag.

    Ollama only auto-resolves ``model`` → ``model:latest``.  If the user
    downloaded ``ministral-3:14b`` (no ``:latest`` alias), asking for
    ``ministral-3`` fails.  This function queries the local server and
    returns the best match:

    * Exact match (name already has a tag)              → return as-is
    * ``bare:latest`` exists                            → ``bare:latest``
    * Exactly one model whose base matches *bare_name*  → that model
    * Multiple matches                                  → first alphabetically
    * No match                                          → return the input unchanged
    """
    local = list_local_models(base_url, timeout=timeout)
    if not local:
        return bare_name

    # Already a fully qualified name?
    if bare_name in local:
        return bare_name

    # Try :latest shortcut
    if f"{bare_name}:latest" in local:
        return f"{bare_name}:latest"

    # Find all models whose base name (before ':') matches
    matches = [m for m in local if m.split(":")[0] == bare_name]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # Prefer :latest if present, otherwise first alphabetically
        for m in matches:
            if m.endswith(":latest"):
                return m
        return sorted(matches)[0]

    return bare_name


def pull_model_streaming(model_name: str, base_url: str = DEFAULT_BASE_URL):
    """Generator that yields progress dicts while pulling a model.

    Each yielded dict has keys like:
        {"status": "pulling ...", "digest": "...", "total": N, "completed": N}
    The final dict typically has {"status": "success"}.

    Raises urllib.error.URLError on connection failure.
    """
    payload = json.dumps({"name": model_name, "stream": True}).encode()
    req = urllib.request.Request(
        f"{base_url}/api/pull",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:  # noqa: S310
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def pull_model_sync(
    model_name: str,
    base_url: str = DEFAULT_BASE_URL,
    progress_callback: Optional[callable] = None,
) -> bool:
    """Pull a model, blocking until complete.

    *progress_callback(pct: float, status: str)* is called periodically
    with a percentage (0-100) and a human-readable status string.
    Returns True on success, False on error.
    """
    try:
        for event in pull_model_streaming(model_name, base_url):
            status = event.get("status", "")
            total = event.get("total", 0)
            completed = event.get("completed", 0)
            pct = (completed / total * 100) if total else 0
            if progress_callback:
                progress_callback(pct, status)
            if status == "success":
                return True
        return True
    except Exception:
        return False
