# import configparser
# import pytest
# from mimicbot_cli import (
#     __app_name__,
#     cli,
#     config,
# )
# import typer
# from typing import Tuple
# from typer.testing import CliRunner, Result
# from configparser import ConfigParser
# from pathlib import Path
# import os
# import pdb

# default_path = Path(config.APP_DIR_PATH)
# default_config = default_path / "config.ini"

# raw_messages_str = """author_id,content,timestamp,channel
# 111111111111111111,8...,2022-07-06 00:35:18.108,test
# 111111111111111111,7...,2022-07-01 01:40:39.608,test
# 111111111111111111,6...,2022-07-01 01:40:37.080,test
# 111111111111111111,5..,2022-07-01 01:40:35.641,test
# 111111111111111111,4..,2022-07-01 01:40:34.404,test
# 111111111111111111,3..,2022-07-01 01:40:33.570,test
# 111111111111111111,2..,2022-07-01 01:40:32.043,test
# 111111111111111111,1...,2022-07-01 01:40:31.337,test
# """

# members_str = """id,name
# 111111111111111111,SomeUser"""


# @pytest.fixture
# def mock_config_path_and_data(tmp_path) -> Tuple[Path, Path]:
#     tmp_config_path = tmp_path / "config.ini"
#     config.init_app(tmp_path)
#     # copy a file from path default_config and paste it to tmp_config_path
#     tmp_config_path.write_text(default_config.read_text())
#     # transform raw_messages_str into a csv file

#     tmp_session_path = tmp_path / "data" / "test_server" / "test_session"
#     if not tmp_session_path.exists():
#         os.makedirs(str(tmp_session_path))
#     (tmp_session_path / "raw_messages.csv").write_text(raw_messages_str)
#     (tmp_session_path / "members.csv").write_text(members_str)
#     return (tmp_config_path, tmp_session_path)

# @pytest.fixture
# def mock_config_path(tmp_path):
#     tmp_config_path = tmp_path / "config.ini"
#     config.init_app(tmp_path)
#     # copy a file from path default_config and paste it to tmp_config_path
#     tmp_config_path.write_text(default_config.read_text())
#     return tmp_config_path


# class TestTrain:
#     # Must have an initialized mimicbot config to the default parameters
#     def test_overfitting(self, mock_config_path, tmp_path):
#         # test that the ROUGE score is higher than .5 after training
#         # with the default parameters
#         runner = CliRunner()
#         result = runner.invoke(cli, ["train", "--dont-save"])
#         assert result.exit_code == 0
#         assert "Training finished" in result.stdout