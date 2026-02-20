from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import requests
from flask import Request
from jose import jwt
from jose.exceptions import JWTError

_JWKS_CACHE: Dict[str, Any] = {"jwks": None, "fetched_at": 0.0}
_JWKS_TTL_SECONDS = 60 * 60  # 1 hour


def _bool_env(name: str, default: bool = False) -> bool:
    v = (os.getenv(name, str(default))).strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def _auth_config() -> Dict[str, str]:
    domain = (os.getenv("AUTH0_DOMAIN") or "").strip()
    audience = (os.getenv("AUTH0_AUDIENCE") or "").strip()
    issuer = (os.getenv("AUTH0_ISSUER") or "").strip()
    if domain and not issuer:
        issuer = f"https://{domain}/"
    return {"domain": domain, "audience": audience, "issuer": issuer}


def _extract_bearer_token(req: Request) -> Optional[str]:
    auth = (req.headers.get("Authorization") or "").strip()
    if not auth or not auth.lower().startswith("bearer "):
        return None
    return auth.split(" ", 1)[1].strip() or None


def _get_jwks(domain: str) -> Dict[str, Any]:
    now = time.time()
    if _JWKS_CACHE["jwks"] is not None and (now - float(_JWKS_CACHE["fetched_at"])) < _JWKS_TTL_SECONDS:
        return _JWKS_CACHE["jwks"]

    url = f"https://{domain}/.well-known/jwks.json"
    resp = requests.get(url, timeout=5)
    resp.raise_for_status()
    jwks = resp.json()

    _JWKS_CACHE["jwks"] = jwks
    _JWKS_CACHE["fetched_at"] = now
    return jwks


def _pick_rsa_key(jwks: Dict[str, Any], kid: str) -> Optional[Dict[str, str]]:
    for k in (jwks.get("keys") or []):
        if k.get("kid") == kid:
            return {"kty": k.get("kty"), "kid": k.get("kid"), "use": k.get("use"), "n": k.get("n"), "e": k.get("e")}
    return None


def get_user_from_request(req: Request) -> Dict[str, str]:
    """
    If AUTH_REQUIRED=true, missing/invalid token raises ValueError.
    Otherwise falls back to anon (dev-friendly).
    """
    cfg = _auth_config()
    required = _bool_env("AUTH_REQUIRED", default=False)

    token = _extract_bearer_token(req)
    if not token:
        if required:
            raise ValueError("Missing Authorization bearer token.")
        return {"user_id": "anon", "user_name": "Anonymous", "user_role": "user"}

    if not cfg["domain"] or not cfg["audience"] or not cfg["issuer"]:
        if required:
            raise ValueError("Auth0 config missing. Set AUTH0_DOMAIN, AUTH0_AUDIENCE, AUTH0_ISSUER.")
        return {"user_id": "anon", "user_name": "Anonymous", "user_role": "user"}

    try:
        header = jwt.get_unverified_header(token)
        kid = header.get("kid")
        if not kid:
            raise ValueError("Invalid token header (kid missing).")

        jwks = _get_jwks(cfg["domain"])
        rsa_key = _pick_rsa_key(jwks, kid)
        if not rsa_key:
            raise ValueError("Signing key not found.")

        claims = jwt.decode(
            token,
            rsa_key,
            algorithms=["RS256"],
            audience=cfg["audience"],
            issuer=cfg["issuer"],
            options={"verify_aud": True, "verify_iss": True},
        )

        user_id = str(claims.get("sub") or "").strip()
        if not user_id:
            raise ValueError("Invalid token (sub missing).")

        user_name = (
            str(claims.get("name") or "").strip()
            or str(claims.get("nickname") or "").strip()
            or str(claims.get("email") or "").strip()
            or "User"
        )

        return {"user_id": user_id, "user_name": user_name, "user_role": "user"}

    except (requests.RequestException, JWTError, ValueError) as e:
        if required:
            raise ValueError(f"Unauthorized: {str(e)}")
        return {"user_id": "anon", "user_name": "Anonymous", "user_role": "user"}

