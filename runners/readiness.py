"""Poll a service until its health probe passes, or the process dies, or time runs out."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any


def wait_until_ready(
    probe: Callable[[], bool],
    *,
    is_alive: Callable[[], bool] | None = None,
    timeout_s: int = 1800,
    stall_timeout_s: int | None = None,
    progress: Callable[[], object] | None = None,
    interval_s: float = 5.0,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
) -> bool:
    """Poll ``probe`` until it returns True; give up if the process dies or time runs out.

    Returns True as soon as ``probe()`` is truthy. If ``is_alive`` is supplied and
    reports False (the server process crashed), returns False immediately. Returns
    False when ``timeout_s`` elapses without a successful probe.
    """
    started_at = now()
    deadline = started_at + timeout_s
    last_progress_at = started_at
    last_progress = progress() if progress is not None else None
    while True:
        if is_alive is not None and not is_alive():
            return False
        if probe():
            return True
        current = now()
        if progress is not None:
            current_progress = progress()
            if current_progress != last_progress:
                last_progress = current_progress
                last_progress_at = current
        if current >= deadline:
            return False
        if stall_timeout_s is not None and current - last_progress_at >= stall_timeout_s:
            return False
        sleep(interval_s)


def make_health_probe(
    container: Any,
    *,
    host: str = "127.0.0.1",
    port: int = 30000,
) -> Callable[[], bool]:
    """Build a probe that curls the SGLang ``/health`` endpoint from inside the container."""
    command = f"curl -sf http://{host}:{port}/health"

    def probe() -> bool:
        return container.exec(command).ok

    return probe
