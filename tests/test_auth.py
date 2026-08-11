"""Kalshi request signing — current official scheme.

timestamp (MILLISECONDS) + HTTP_METHOD + full /trade-api/v2 path (no query)
signed with RSA-PSS/SHA-256 (PSS.MAX_LENGTH salt).
"""

from __future__ import annotations

import base64
import re

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey

from weadge.adapters.kalshi.auth import API_PATH_PREFIX, _message, sign_headers

RSA_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
PEM = RSA_KEY.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.TraditionalOpenSSL,
    encryption_algorithm=serialization.NoEncryption(),
).decode()


def _verify(signature_b64: str, msg: str, key: RSAPrivateKey = RSA_KEY) -> bool:
    try:
        key.public_key().verify(
            base64.b64decode(signature_b64),
            msg.encode(),
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )
        return True
    except Exception:  # verification failure means wrong signature
        return False


class TestMessage:
    def test_message_format_is_ms_timestamp_method_path(self) -> None:
        assert _message("1786400000000", "GET", "/trade-api/v2/series/KXHIGHNY") == (
            "1786400000000GET/trade-api/v2/series/KXHIGHNY"
        )

    def test_method_is_upper_cased(self) -> None:
        assert _message("1", "get", "/trade-api/v2/x") == "1GET/trade-api/v2/x"

    def test_query_never_participates(self) -> None:
        """The signed path is the path WITHOUT query parameters."""
        assert _message("1", "GET", "/trade-api/v2/markets") != _message(
            "1", "GET", "/trade-api/v2/markets?limit=200"
        )


class TestSignHeaders:
    def test_timestamp_is_milliseconds(self) -> None:
        headers = sign_headers("key", PEM, method="GET", path="/trade-api/v2/series/KXHIGHNY")
        ts = headers["Kalshi-Access-Timestamp"]
        assert re.fullmatch(r"\d{13}", ts)  # epoch ms = 13 digits

    def test_signature_verifies_over_ms_timestamp_method_path(self) -> None:
        headers = sign_headers("key", PEM, method="GET", path="/trade-api/v2/series/KXHIGHNY")
        msg = _message(headers["Kalshi-Access-Timestamp"], "GET", "/trade-api/v2/series/KXHIGHNY")
        assert _verify(headers["Kalshi-Access-Signature"], msg)

    def test_signature_binds_method_and_path(self) -> None:
        h1 = sign_headers("key", PEM, method="GET", path="/trade-api/v2/markets")
        h2 = sign_headers("key", PEM, method="POST", path="/trade-api/v2/markets")
        h3 = sign_headers("key", PEM, method="GET", path="/trade-api/v2/events")
        assert h1["Kalshi-Access-Signature"] != h2["Kalshi-Access-Signature"]
        assert h1["Kalshi-Access-Signature"] != h3["Kalshi-Access-Signature"]

    def test_historical_path_is_signed_with_prefix(self) -> None:
        headers = sign_headers("key", PEM, method="GET", path="/trade-api/v2/historical/markets")
        msg = _message(headers["Kalshi-Access-Timestamp"], "GET", "/trade-api/v2/historical/markets")
        assert _verify(headers["Kalshi-Access-Signature"], msg)

    def test_path_without_prefix_rejected(self) -> None:
        with pytest.raises(ValueError, match="trade-api/v2"):
            sign_headers("key", PEM, method="GET", path="/markets")

    def test_missing_credentials_raise(self, monkeypatch) -> None:
        monkeypatch.delenv("KALSHI_API_KEY", raising=False)
        monkeypatch.delenv("KALSHI_API_SECRET", raising=False)
        with pytest.raises(RuntimeError, match="KALSHI_API_KEY"):
            sign_headers(method="GET", path=API_PATH_PREFIX)

    def test_credentials_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("KALSHI_API_KEY", "env-key")
        monkeypatch.setenv("KALSHI_API_SECRET", PEM)
        headers = sign_headers(method="GET", path="/trade-api/v2/series/KXHIGHNY")
        assert headers["Kalshi-Access-Key"] == "env-key"
