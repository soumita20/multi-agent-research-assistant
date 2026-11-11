import pytest

from app_settings import PROJECT_SETTINGS, load_project_settings, provider_for_driver


def test_model_catalog_is_loaded_from_project_configuration():
    assert list(PROJECT_SETTINGS["providers"]) == ["Ollama", "Groq", "Gemini"]
    assert PROJECT_SETTINGS["providers"]["Gemini"]["default_model"] == "gemini-3.8-flash"
    assert provider_for_driver(PROJECT_SETTINGS, "google") == "Gemini"


def test_config_loader_rejects_unlisted_default_model(tmp_path):
    config_path = tmp_path / "invalid.toml"
    config_path.write_text(
        """[runtime]
default_family = "Custom"

[providers.Custom]
driver = "ollama"
default_model = "missing-model"
models = [{ id = "available-model", label = "Available" }]
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="default_model must be listed"):
        load_project_settings(config_path)
