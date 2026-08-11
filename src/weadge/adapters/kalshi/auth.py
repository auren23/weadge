"""Kalshi request signing (RSA-PSS/SHA-256, per Kalshi docs).

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


def sign_headers(api_key: str | None = None, api_secret: str | None = None) -> dict[str, str]:
    """Return Kalshi authentication headers for the current timestamp."""
    key = api_key or os.environ.get("KALSHI_API_KEY")
    secret = api_secret or os.environ.get("KALSHI_API_SECRET")
    if not key or not secret:
        raise RuntimeError(
            "Kalshi auth requires KALSHI_API_KEY and KALSHI_API_SECRET env vars"
        )
    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding, rsa
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "kalshi auth needs the 'cryptography' package: uv add cryptography"
        ) from exc

    timestamp = str(int(time.time()))
    msg = f"{timestamp}{key}"
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


__all__: list[Any] = ["get_credentials_from_env", "sign_headers"]
