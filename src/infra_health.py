from __future__ import annotations

import socket
from urllib.parse import urlparse


def _host_port_from_url(url: str) -> tuple[str, int] | None:
    parsed = urlparse(url)
    if not parsed.hostname or not parsed.port:
        return None
    return parsed.hostname, parsed.port


def tcp_available(host: str, port: int, timeout_seconds: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            return True
    except OSError:
        return False


def service_probe(name: str, url_or_host: str, port: int | None = None) -> dict[str, object]:
    if port is None:
        parsed = _host_port_from_url(url_or_host)
        if parsed is None:
            return {"name": name, "configured": bool(url_or_host), "tcp_available": None}
        host, resolved_port = parsed
    else:
        host, resolved_port = url_or_host, port

    return {
        "name": name,
        "host": host,
        "port": resolved_port,
        "tcp_available": tcp_available(host, resolved_port),
    }
