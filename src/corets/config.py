"""Configuration contains environment-variable names, never credentials."""
from __future__ import annotations
import copy
import os
from pathlib import Path
from urllib.parse import urlsplit
from .io import read_json

DEFAULTS = {
    "questions_per_repository": 4,
    "concurrency": 16,
    "max_author_attempts_per_repository": 24,
    "max_evidence_packs": 32,
    "evidence_characters": 24000,
    "source_excerpt_characters": 36000,
    "max_iterations": 25,
    "max_context_tokens": 32768,
    "context_warning_threshold": 0.825,
    "episode_timeout_seconds": 3600,
    "minimum_free_gib": 5,
    "tools_image": "corets-tools:93ac6a4",
    "seed": 42,
    "models": {},
    "quality_thresholds": {"correctness": 9, "completeness": 9, "relevance": 8, "clarity": 7, "reasoning": 8, "grounding": 8},
}
ROLES = ("scanner", "author", "teacher", "reviewer")


def load_config(path: Path) -> dict:
    given = read_json(path)
    unknown = set(given) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    config = copy.deepcopy(DEFAULTS)
    config.update(given)
    for field in ("questions_per_repository", "concurrency", "max_author_attempts_per_repository", "max_evidence_packs", "max_iterations", "max_context_tokens", "episode_timeout_seconds"):
        if type(config[field]) is not int or config[field] < 1:
            raise ValueError(f"{field} must be a positive integer")
    if config["max_author_attempts_per_repository"] < config["questions_per_repository"]:
        raise ValueError("The author-attempt budget is smaller than the question target")
    if not 0 < config["context_warning_threshold"] < 1:
        raise ValueError("context_warning_threshold must be between zero and one")
    if set(config["models"]) != set(ROLES):
        raise ValueError("Configure scanner, author, teacher and reviewer models")
    for role in ROLES:
        model = config["models"][role]
        if any(k in model for k in ("api_key", "token", "password", "base_url")):
            raise ValueError("Use api_key_env/base_url_env instead of embedding endpoints or credentials")
        for key in ("model", "api_key_env", "base_url_env"):
            if not isinstance(model.get(key), str) or not model[key]:
                raise ValueError(f"Missing models.{role}.{key}")
        if not isinstance(model.get("parameters", {}), dict):
            raise ValueError("Model parameters must be an object")
        if {"model", "messages", "tools", "stream", "api_key"} & set(model.get("parameters", {})):
            raise ValueError("Reserved model parameters cannot be overridden")
    scores = config["quality_thresholds"]
    if set(scores) != set(DEFAULTS["quality_thresholds"]) or any(type(v) is not int or not 1 <= v <= 10 for v in scores.values()):
        raise ValueError("Quality thresholds must define six integer scores in 1..10")
    return config


def credentials(profile: dict) -> tuple[str, str]:
    base = os.environ.get(profile["base_url_env"], "").rstrip("/")
    key = os.environ.get(profile["api_key_env"], "")
    if not base or not key:
        raise ValueError(f"Set {profile['base_url_env']} and {profile['api_key_env']}")
    parsed = urlsplit(base)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("API base must be an HTTP(S) URL without embedded credentials, query, or fragment")
    return base, key
