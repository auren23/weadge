"""Kalshi request signing (RSA-PSS/SHA-256, per Kalshi docs).

Current official scheme (2026):
    timestamp = unix time in MILLISECONDS
    message   = f"{timestamp}{HTTP_METHOD}{full_path}"
    full_path = "/trade-api/v2/..." — the request path as sent, WITHOUT
                query parameters (query strings never participate in the
                signature)
    signature = RSA-PSS/SHA-256 over the message (PSS.MAX_LENGTH salt)

Keys come from environment variables, never from code or config:
    KALSHI_API_KEY      — the account key id
    KALSHI_API_SECRET   — the PEM-encoded RSA private key

The `cryptography` package is imported lazily so public-endpoint users never
pay for it.
"""

from __future__ import annotations

import base64
import os
import time
from typing import Any

API_PATH_PREFIX = "/trade-api/v2"


def _message(timestamp_ms: str, method: str, path: str) -> str:
    """timestamp + METHOD + path — the exact bytes Kalshi signs.

    `path` must be the full request path INCLUDING the /trade-api/v2 prefix
    and EXCLUDING any query string (e.g. "/trade-api/v2/historical/markets").
    """
    return f"{timestamp_ms}{method.upper()}{path}"


def sign_headers(
    api_key: str | None = None,
    api_secret: str | None = None,
    *,
    method: str = "GET",
    path: str = API_PATH_PREFIX,
) -> dict[str, str]:
    """Return Kalshi authentication headers for the current millisecond."""
    key = api_key or os.environ.get("KALSHI_API_KEY")
    secret = api_secret or os.environ.get("KALSHI_API_SECRET")
    if not key or not secret:
        raise RuntimeError(
            "Kalshi auth requires KALSHI_API_KEY and KALSHI_API_SECRET env vars"
        )
    if not path.startswith(API_PATH_PREFIX):
        raise ValueError(
            f"signed path must start with {API_PATH_PREFIX!r} and carry no query, got {path!r}"
        )
    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding, rsa
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "kalshi auth needs the 'cryptography' package: uv add cryptography"
        ) from exc

    timestamp = str(int(time.time() * 1000))  # milliseconds, per official scheme
    msg = _message(timestamp, method, path)
    loaded = serialization.load_pem_private_key(secret.encode(), password=None)
    # Kalshi signs with an RSA private key; other key types cannot sign this scheme
    if not isinstance(loaded, rsa.RSAPrivateKey):
        raise RuntimeError("KALSHI_API_SECRET must be an RSA private key (PEM)")
    signature = loaded.sign(  # type: ignore[attr-defined]
        msg.encode(),
        padding.PSS(  # type: ignore[call-arg]
            mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256(),
    )
    return {
        "Kalshi-Access-Key": key,
        "Kalshi-Access-Timestamp": timestamp,
        "Kalshi-Access-Signature": base64.b64encode(signature).decode(),
    }


def get_credentials_from_env() -> tuple[str | None, str | None]:
    """Read credentials from the environment (never prints them)."""
    return os.environ.get("KALSHI_API_KEY"), os.environ.get("KALSHI_API_SECRET")


__all__: list[Any] = ["API_PATH_PREFIX", "get_credentials_from_env", "sign_headers"]
