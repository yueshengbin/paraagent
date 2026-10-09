"""Keep credentials out of public metadata and diagnostic configuration logs."""

from collections.abc import Mapping
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def validate_public_url(url, label="base_url"):
    """Require a service URL whose identity can safely be recorded publicly."""
    try:
        parsed = urlsplit(url)
        valid = (
            parsed.scheme in {"http", "https"}
            and parsed.hostname
            and parsed.username is None
            and parsed.password is None
            and not parsed.query
            and not parsed.fragment
        )
        parsed.port
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError(f"{label} must be an HTTP(S) URL without credentials, query parameters or fragments")
    return url


def redact_config(value):
    """Return a log-safe copy; environment values are always treated as private."""
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            name = str(key).lower().replace("-", "_")
            sensitive = (
                name in {"key", "token"}
                or name.endswith("_token")
                or any(part in name for part in (
                    "api_key", "apikey", "secret", "password", "passwd",
                    "credential", "authorization", "private_key", "access_key", "toolbench_key",
                ))
            )
            if name == "env_vars" and isinstance(item, Mapping):
                result[key] = {env_key: "[REDACTED]" for env_key in item}
            elif sensitive:
                result[key] = "[REDACTED]"
            else:
                result[key] = redact_config(item)
        return result
    if isinstance(value, (list, tuple)):
        return [redact_config(item) for item in value]
    if isinstance(value, str) and value.startswith(("http://", "https://")):
        try:
            parsed = urlsplit(value)
            query = urlencode([(key, "[REDACTED]") for key, _ in parse_qsl(parsed.query, keep_blank_values=True)])
            return urlunsplit((parsed.scheme, parsed.netloc.rsplit("@", 1)[-1], parsed.path, query, ""))
        except ValueError:
            return "[REDACTED URL]"
    return value
