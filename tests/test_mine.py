import pytest
from mimicbot import (
    __app_name__,
    cli,
    config,
)
import typer
from typer.testing import CliRunner, Result
from configparser import ConfigParser
from pathlib import Path 

# default_path = Path(config.APP_DIR_PATH)
# default_config = default_path / "config.ini"

# @pytest.fixture
# def mock_config_path(tmp_path):
#     tmp_config_path = tmp_path / "config.ini"
#     config.init_app(tmp_path)
#     # copy a file from path default_config and paste it to tmp_config_path
#     tmp_config_path.write_text(default_config.read_text())
#     return tmp_config_path


class TestMine:
    def test_base(self):
        assert True
    # # Must have an initialized mimicbot config to the default parameters
    # def test_successful_mine(self, mock_config_path, tmp_path):
    #     parsed_config = ConfigParser()
    #     parsed_config.read(str(mock_config_path))
    #     runner = CliRunner()
    #     result = runner.invoke(cli.app, ["mine", "-ap", tmp_path])

    #     data_path = tmp_path / "data" / parsed_config.get("discord", "guild") / parsed_config.get("general", "session")
    #     messages_path = data_path / "raw_messages.csv"

    #     assert "Successfully mined data." in result.stdout
    #     assert result.exit_code == 0
    #     assert messages_path.exists()
    
    # def test_failed_mine_api_key(self, mock_config_path, tmp_path):
    #     parsed_config = ConfigParser()
    #     parsed_config.read(str(mock_config_path))
    #     parsed_config.set("discord", "api_key", "xxFAKExx")
    #     with open(str(mock_config_path), "w") as config_file:
    #         parsed_config.write(config_file)

    #     runner = CliRunner()
    #     result = runner.invoke(cli.app, ["mine", "-ap", tmp_path])

    #     assert "Error: API_KEY_ERROR" in result.stdout
    #     assert result.exit_code == 1