"""Load configurable application and model-catalog settings from TOML."""

import os
import tomllib
from pathlib import Path
from typing import Any


def load_project_settings(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path or os.getenv("APP_CONFIG_FILE", Path(__file__).with_name("config.toml")))
    with config_path.open("rb") as config_file:
        settings = tomllib.load(config_file)

    if "runtime" not in settings or "providers" not in settings:
        raise ValueError(f"Application config is missing required sections: {config_path}")
    if settings["runtime"]["default_family"] not in settings["providers"]:
        raise ValueError("runtime.default_family must name an entry in providers")
    for family, provider in settings["providers"].items():
        if not provider.get("driver") or not provider.get("models"):
            raise ValueError(f"Provider {family!r} must define a driver and at least one model")
        model_ids = {model["id"] for model in provider["models"]}
        if provider.get("default_model") not in model_ids:
            raise ValueError(f"Provider {family!r} default_model must be listed in models")
    return settings


PROJECT_SETTINGS = load_project_settings()


def provider_for_driver(settings: dict[str, Any], driver: str) -> str | None:
    return next(
        (family for family, provider in settings["providers"].items()
         if provider["driver"] == driver),
        None,
    )
