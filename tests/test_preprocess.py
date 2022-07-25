import configparser
import pytest
from mimicbot import (
    __app_name__,
    cli,
    config,
)
import typer
from typing import Tuple
from typer.testing import CliRunner, Result
from configparser import ConfigParser
from pathlib import Path
import os
import pdb

raw_messages_str = """author_id,content,timestamp,channel
111111111111111111,8...,2022-07-06 00:35:18.108,test
111111111111111111,7...,2022-07-01 01:40:39.608,test
111111111111111111,6...,2022-07-01 01:40:37.080,test
111111111111111111,5..,2022-07-01 01:40:35.641,test
111111111111111111,4..,2022-07-01 01:40:34.404,test
111111111111111111,3..,2022-07-01 01:40:33.570,test
111111111111111111,2..,2022-07-01 01:40:32.043,test
111111111111111111,1...,2022-07-01 01:40:31.337,test
"""

members_str = """id,name
111111111111111111,SomeUser"""


@pytest.fixture
def mock_config_path_and_data(tmp_path) -> Tuple[Path, Path]:
    tmp_config_path = tmp_path / "config.ini"
    config.init_app(tmp_path)

    # transform raw_messages_str into a csv file
    tmp_session_path = tmp_path / "data" / "test_server" / "test_session"
    if not tmp_session_path.exists():
        os.makedirs(str(tmp_session_path))
    (tmp_session_path / "raw_messages.csv").write_text(raw_messages_str)
    (tmp_session_path / "members.csv").write_text(members_str)
    return (tmp_config_path, tmp_session_path)


class TestPreprocess:
    # Must have an initialized mimicbot config to the default parameters
    def test_standard_preprocess(self, mock_config_path_and_data):
        parsed_config = ConfigParser()
        parsed_config.read(str(mock_config_path_and_data[0]))
        parsed_config.set("discord", "target_user", "SomeUser")
        parsed_config.set("training", "context_window", "")
        parsed_config.set("training", "context_length", "6")
        parsed_config.set("training", "test_perc", "0.1")
        with open(str(mock_config_path_and_data[0]), "w") as config_file:
            parsed_config.write(config_file)
        runner = CliRunner()
        result = runner.invoke(
            cli.app, ["preprocess", "--session-path", mock_config_path_and_data[1]])

        train_path = mock_config_path_and_data[1] / \
            "training_data" / "train.csv"

        assert "Data is ready for training. You can find it here" in result.stdout
        assert result.exit_code == 0
        assert train_path.exists()
        # ensure there are only 3 lines in trin_path
        assert len(train_path.read_text().split("\n")) == 4

    # def test_extrapolated_preprocess(self, mock_config_path_and_data):
    #     parsed_config = ConfigParser()
    #     parsed_config.read(str(mock_config_path_and_data[0]))
    #     parsed_config.set("discord", "target_user", "SomeUser")
    #     parsed_config.set("training", "context_window", "6")
    #     parsed_config.set("training", "context_length", "2")
    #     parsed_config.set("training", "test_perc", "0.1")
    #     with open(str(mock_config_path_and_data[0]), "w") as config_file:
    #         parsed_config.write(config_file)
    #     runner = CliRunner()
    #     result = runner.invoke(
    #         cli.app, ["preprocess", "--session-path", mock_config_path_and_data[1]])

    #     train_path = mock_config_path_and_data[1] / \
    #         "training_data" / "train.csv"

    #     assert "Data is ready for training. You can find it here" in result.stdout
    #     assert result.exit_code == 0
    #     assert train_path.exists()
    #     # ensure there are only 3 lines in trin_path
    #     assert len(train_path.read_text().split("\n")) == 32
