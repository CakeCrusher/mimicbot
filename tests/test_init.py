import pdb
from pathlib import Path
import pytest
from mimicbot import (
    __app_name__,
    cli,
    utils,
)
import typer
from typer.testing import CliRunner, Result
from configparser import ConfigParser

runner = CliRunner()


def file_success_assertions(
    result: Result,
    config_path: Path,
    # session can be a bool or a string
    session: str or bool,
    discord_api_key: str,
    huggingface_api_key: str,
):
    assert "Successfully initialized mimicbot." in result.stdout

    assert config_path.exists()

    config = ConfigParser()
    config.read(str(config_path))

    assert config.has_section("general")
    if session:
        assert config.get("general", "session") == session
    else:
        assert config.get("general", "session") == str(utils.datetime_str())

    assert config.has_section("discord")
    assert config.get("discord", "api_key") == discord_api_key

    assert config.has_section("huggingface")
    assert config.get("huggingface", "api_key") == huggingface_api_key


@pytest.mark.parametrize(
    "session, data_path, discord_api_key, discord_guild, discord_target_user, huggingface_api_key, huggingface_model_name",
    [
        pytest.param(
            "test_session",
            "",
            "dxxx-xxxx",
            "test_guild",
            "test_user",
            "hxxx-xxxx",
            "mimicbot"
        ),
        pytest.param(
            "test_session",
            "",
            "dxxx-xxxx",
            "test_guild",
            "test_user",
            "hxxx-xxxx",
            "mimicbot"
        ),
    ]
)
class TestInit:
    def test_abort(self, session, data_path, discord_api_key, discord_guild, discord_target_user, huggingface_api_key, huggingface_model_name):
        result = runner.invoke(cli.app, ["init"])
        app_path = Path(typer.get_app_dir(__app_name__))
        assert f"Path to store data [{str(app_path)}]" in result.stdout or f"[{str(app_path)}] config already exists." in result.stdout
        assert "Aborted!" in result.stdout

    def test_create_file(self, tmp_path, session, data_path, discord_api_key, discord_guild, discord_target_user, huggingface_api_key, huggingface_model_name):
        app_path = tmp_path / "mimicbot"
        data_path = app_path / "data"
        config_path = app_path / "config.ini"

        result = runner.invoke(
            cli.app, ["init", "--app-path", str(app_path)],
            input=f"{str(data_path)}\n{discord_api_key}\n{discord_guild}\n{discord_target_user}\n{huggingface_api_key}\n{huggingface_model_name}\n"
        )

        file_success_assertions(result, config_path,
                                False, discord_api_key, huggingface_api_key)

    def test_create_file_forced(self, tmp_path, session, data_path, discord_api_key, discord_guild, discord_target_user, huggingface_api_key, huggingface_model_name):
        app_path = tmp_path / "mimicbot"
        data_path = app_path / "data"
        config_path = app_path / "config.ini"

        result = runner.invoke(
            cli.app, [
                "init",
                "--app-path", str(app_path),
                "--data-path", str(data_path),
                "--session", session,
                "--discord-api-key", discord_api_key,
                "--discord-guild", discord_guild,
                "--discord-target-user", discord_target_user,
                "--huggingface-api-key", huggingface_api_key,
                "--huggingface-model-name", huggingface_model_name,
            ]
        )

        file_success_assertions(result, config_path,
                                session, discord_api_key, huggingface_api_key)
