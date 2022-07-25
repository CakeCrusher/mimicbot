import pdb
from pathlib import Path
import pytest
from mimicbot import (
    __app_name__,
    cli,
    utils,
    config,
)
import typer
from typer.testing import CliRunner, Result
from configparser import ConfigParser

runner = CliRunner()


def file_success_assertions(
    result: Result,
    config_path: Path,
    session: str or bool,
    discord_api_key: str,
    huggingface_api_key: str,
    context_length: int or bool,
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

    assert config.has_section("training")
    if context_length:
        assert config.get("training", "context_length") == str(context_length)
    else:
        assert config.get("training", "context_length") == str(
            2)  # default value


@pytest.mark.parametrize(
    "session, data_path, discord_api_key, discord_guild, discord_target_user, huggingface_api_key, huggingface_model_name, context_length, context_window, test_perc",
    [
        pytest.param(
            "test_session",
            "",
            "dxxx-xxxx",
            "test_guild",
            "test_user",
            "hxxx-xxxx",
            "mimicbot-test_session",
            3,
            5,
            0.1,
        ),
        pytest.param(
            "test_session",
            "",
            "dxxx-xxxx",
            "test_guild",
            "test_user",
            "hxxx-xxxx",
            "mimicbot-test_session",
            3,
            5,
            0.1,
        ),
    ]
)
class TestInit:
    # def test_abort(self, tmp_path, session, data_path, discord_api_key, discord_guild, discord_target_user, huggingface_api_key, huggingface_model_name, context_length, context_window, test_perc):
    #     # TODO: must figure out how to abort thr program without "\n"
    #     result = runner.invoke(cli.app, ["init", "--app-path", tmp_path], input="\x03")
    #     assert f"Session name" in result.stdout
    #     assert "Aborted!" in result.stdout
    def test_no_override(self, tmp_path, session, data_path, discord_api_key, discord_guild, discord_target_user, huggingface_api_key, huggingface_model_name, context_length, context_window, test_perc):
        app_path = tmp_path / "mimicbot"
        config.init_app(app_path)
        result = runner.invoke(cli.app, ["init", "--app-path", app_path], input="n\n")
        assert f"Config already exists in [{str(app_path)}]" in result.stdout
        assert "Aborted!" in result.stdout

    def test_create_file(self, tmp_path, session, data_path, discord_api_key, discord_guild, discord_target_user, huggingface_api_key, huggingface_model_name, context_length, context_window, test_perc):
        app_path = tmp_path / "mimicbot"
        data_path = app_path / "data"
        config_path = app_path / "config.ini"
        print("!!SESSION:", session)

        result = runner.invoke(
            cli.app, ["init", "--app-path", str(app_path)],
            input=f"{str(session)}\n\n{discord_api_key}\n{discord_guild}\n{discord_target_user}\n{huggingface_api_key}\n{huggingface_model_name}\ny\n"
        )

        file_success_assertions(result, config_path,
                                session, discord_api_key, huggingface_api_key, False)

    def test_create_file_with_training(self, tmp_path, session, data_path, discord_api_key, discord_guild, discord_target_user, huggingface_api_key, huggingface_model_name, context_length, context_window, test_perc):
        app_path = tmp_path / "mimicbot"
        data_path = app_path / "data"
        config_path = app_path / "config.ini"

        result = runner.invoke(
            cli.app, ["init", "--app-path", str(app_path)],
            input=f"{str(session)}\n\n{discord_api_key}\n{discord_guild}\n{discord_target_user}\n{huggingface_api_key}\n{huggingface_model_name}\nn\ny\n{context_length}\n{context_window}\n{test_perc}\n"
        )

        file_success_assertions(result, config_path,
                                session, discord_api_key, huggingface_api_key, context_length)

    def test_create_file_forced(self, tmp_path, session, data_path, discord_api_key, discord_guild, discord_target_user, huggingface_api_key, huggingface_model_name, context_length, context_window, test_perc):
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
                                session, discord_api_key, huggingface_api_key, False)
