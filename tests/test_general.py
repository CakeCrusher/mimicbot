import pdb
from pathlib import Path
import pytest
from mimicbot import (
    __app_name__,
    cli,
)
import typer
from typer.testing import CliRunner
from configparser import ConfigParser

runner = CliRunner()


@pytest.mark.parametrize(
    "session, app_path, discord_api_key, discord_guild, discord_target_user, huggingface_api_key",
    [
        pytest.param(
            "test_session",
            "",
            "dxxx-xxxx",
            "test_guild",
            "test_user",
            "hxxx-xxxx",
        ),
        pytest.param(
            "test_session",
            "",
            "dxxx-xxxx",
            "test_guild",
            "test_user",
            "hxxx-xxxx",
        ),
    ]
)

class TestInit:
    def test_abort(self, session, app_path, discord_api_key, discord_guild, discord_target_user, huggingface_api_key):
        result = runner.invoke(cli.app, ["init"])
        assert f"Path to store mimicbot data [{str(Path(typer.get_app_dir(__app_name__)))}]" in result.stdout
        assert "Aborted!" in result.stdout


    def test_create_file(self, tmp_path, session, app_path, discord_api_key, discord_guild, discord_target_user, huggingface_api_key):
        app_path = tmp_path / "mimicbot"
        config_path = app_path / "config.ini"

        result = runner.invoke(
            cli.app, [
                "init",
                "--app-path", str(app_path),
                "--session", session,
                "--discord-api-key", discord_api_key,
                "--discord-guild", discord_guild,
                "--discord-target-user", discord_target_user,
                "--huggingface-api-key", huggingface_api_key
            ]
        )
        assert "Successfully initialized mimicbot." in result.stdout

        assert config_path.exists()

        config = ConfigParser()
        config.read(str(config_path))

        assert config.has_section("general")
        assert config.get("general", "session") == session

        assert config.has_section("discord")
        assert config.get("discord", "api_key") == discord_api_key

        assert config.has_section("huggingface")
        assert config.get("huggingface", "api_key") == huggingface_api_key
