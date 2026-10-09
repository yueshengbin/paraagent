"""ToolBench client for explicitly selected service endpoints."""

from urllib.parse import urlsplit
import math
import os

import requests

from paraagent.paraact.naming import change_name, standardize


class ToolBenchServiceError(RuntimeError):
    pass


class ToolBenchServiceClient:
    def __init__(self, service_url: str, toolbench_key: str | None, backend: str = "live", *,
                 connect_timeout: float | None = None, read_timeout: float | None = None):
        if backend not in {"live", "virtual", "mirrorapi"}:
            raise ValueError(f"Unknown ToolBench service backend: {backend}")
        parsed = urlsplit(service_url or "")
        endpoint = "rapidapi" if backend == "live" else "virtual"
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path.rstrip("/").rsplit("/", 1)[-1] != endpoint
        ):
            raise ValueError(f"TOOLBENCH_SERVICE_URL must be an HTTP(S) /{endpoint} URL without credentials or query parameters")
        if backend == "live" and not toolbench_key:
            raise ValueError("TOOLBENCH_KEY is required for the live ToolBench backend")
        self.service_url = service_url
        self.backend = backend
        self._toolbench_key = toolbench_key or ""
        self.timeout = (
            self._timeout(connect_timeout, "TOOLBENCH_CONNECT_TIMEOUT", 15),
            self._timeout(read_timeout, "TOOLBENCH_READ_TIMEOUT", 15 if backend == "live" else 300),
        )

    @staticmethod
    def _timeout(value, variable, default):
        value = os.environ.get(variable, default) if value is None else value
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{variable} must be a finite positive number of seconds") from None
        if isinstance(value, bool) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError(f"{variable} must be a finite positive number of seconds")
        return seconds

    def call(self, function: dict, tool_input: dict, strip: str) -> dict:
        name = function["name"]
        if " : " not in name:
            raise ToolBenchServiceError("Tool catalog entry has no ToolBench tool/API separator")
        tool_name, api_name = name.split(" : ", 1)
        payload = {
            "category": function["category"],
            "tool_name": standardize(tool_name),
            "api_name": change_name(standardize(api_name)),
            "tool_input": tool_input,
            "strip": strip,
            "toolbench_key": self._toolbench_key,
        }
        try:
            response = requests.post(
                self.service_url,
                json=payload,
                headers={"toolbench_key": self._toolbench_key},
                proxies={"http": "", "https": ""},
                timeout=self.timeout,
            )
        except requests.exceptions.Timeout:
            raise
        except requests.exceptions.RequestException as exc:
            raise ToolBenchServiceError("ToolBench service request failed") from exc
        if response.status_code != 200:
            raise ToolBenchServiceError(f"ToolBench service returned HTTP {response.status_code}")
        try:
            result = response.json()
        except ValueError as exc:
            raise ToolBenchServiceError("ToolBench service returned non-JSON content") from exc
        if not isinstance(result, dict) or "error" not in result or "response" not in result:
            raise ToolBenchServiceError("ToolBench service returned an invalid response envelope")
        return result
