import configparser
import datetime
import requests
from pathlib import Path
from types import NoneType
from typing import Tuple
import typer
from mimicbot import config, __app_name__, types
from collections.abc import Callable
import json
import numpy as np
import pandas as pd

APP_DIR_PATH = Path(typer.get_app_dir(__app_name__))


def delete_folder(pth):
    for sub in pth.iterdir():
        if sub.is_dir():
            delete_folder(sub)
        else:
            sub.unlink()
    pth.rmdir()


def ensure_app_path(app_path: Path) -> Path:
    while not app_path.exists():
        typer.secho(f"Path [{app_path}] does not exist.", fg=typer.colors.RED)
        app_path: str = typer.prompt(
            "Path to mimicbot data", default=str(config.APP_DIR_PATH))
        app_path = Path(app_path)

    return app_path


def app_path_verifier(app_path_str: str) -> None:
    # callback is called twice for some reason
    if not type(app_path_str) == str:
        return config.APP_DIR_PATH
    app_path = Path(app_path_str)
    if (app_path / "config.ini").exists():
        typer.confirm(
            typer.style(
                f"\nConfig already exists in [{app_path_str}] . Do you want to overwrite it?\n", fg=typer.colors.YELLOW),
            False,
            abort=True,
        )
        # # rewrite the config file
        # config_parser = configparser.ConfigParser()
        # with open(str(app_path / "config.ini"), "w") as config_file:
        #     config_parser.write(config_file)

    return app_path_str


def path_verifier(param: typer.CallbackParam, path_str: str) -> Path:
    path = Path(path_str)
    while not path.exists():
        typer.secho(f"Path ({path}) does not exist.", fg=typer.colors.RED)
        path = typer.prompt(f"Enter new {param.name}")
        path = Path(path)
    return path


def datetime_str():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")


def callback_config(app_path: Path = APP_DIR_PATH) -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config_path = app_path / "config.ini"
    try:
        config.read(str(config_path))
    except:
        raise FileNotFoundError
    # if callback:
    #     config = callback(config)
    with open(str(config_path), "w") as config_file:
        config.write(config_file)
    return config


def current_config(category: str, value: str, default: str = None, app_path: Path = APP_DIR_PATH) -> str or None:
    config = configparser.ConfigParser()
    try:
        config.read(str(app_path / "config.ini"))
        return config.get(category, value)
    except:
        return default


def session_path(config: configparser.ConfigParser) -> Path:
    GUILD = config.get("discord", "guild")
    DATA_PATH = config.get("general", "data_path")
    SESSION_NAME = config.get("general", "session")
    SESSION_DATA_PATH = Path(DATA_PATH) / Path(GUILD) / Path(SESSION_NAME)
    return SESSION_DATA_PATH


def add_model_save(app_path, model_save: types.ModelSave):
    config_parser = configparser.ConfigParser()
    config_path = app_path / "config.ini"
    try:
        config_parser.read(str(config_path))
    except:
        raise FileNotFoundError
    current_saves = config_parser.get("huggingface", "model_saves")
    current_saves = json.loads(current_saves)
    new_saves = json.dumps([model_save] + current_saves)
    config_parser.set("huggingface", "model_saves", new_saves)
    with open(str(config_path), "w") as config_file:
        config_parser.write(config_file)


def prompt_model_save() -> int:
    config_parser = callback_config()
    model_saves: list[types.ModelSave] = json.loads(
        config_parser.get("huggingface", "model_saves"))
    models_string = ""
    for idx, model_save in enumerate(model_saves):
        url = model_save["url"]
        models_string += f"({idx}) {url}\n"
    model_idx = ""
    while type(model_idx) != int:
        model_idx = typer.prompt(
            "\nModel to run bot on:\n" + models_string + "Enter number of model",
            default=f"0",
        )
        try:
            model_idx = int(model_idx)
            if abs(model_idx) >= len(model_saves):
                model_idx = ""
                assert False
        except:
            pass

        if type(model_idx) != int:
            typer.secho(
                "The number you entered does not match any model.", fg=typer.colors.RED)
    return model_idx


def standardize_data(messages: pd.DataFrame, members: pd.DataFrame, author_id_column: str, content_column: str, skip_naming: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    standard_messages = pd.DataFrame(columns=["author_id", "content"], data=messages[[
                                     author_id_column, content_column]].values)
    missing_members = list(
        set(messages[author_id_column].unique()) - set(members["id"].unique()))
    while bool(len(missing_members)):
        random_member_name = f'Human-{np.random.randint(100, 999)}'
        if skip_naming:
            member_name = random_member_name
        else:
            member_name = typer.prompt(
                f'\nEnter name for member with id ({missing_members[0]})', default=random_member_name)
        members = pd.concat(
            [
                members,
                pd.DataFrame(columns=['id', 'name'], data=[
                             [missing_members[0], member_name]])
            ],
            ignore_index=True
        )

        missing_members = list(
            set(messages[author_id_column].unique()) - set(members["id"].unique()))

    return (standard_messages, members)

def save_standardized_data(messages_path: str, members_path: str, output_dir: str, author_id_column: str, content_column: str) -> Path:
    messages = pd.read_csv(messages_path)
    try:
        members = pd.read_csv(members_path)
    except:
        members = pd.DataFrame(columns=['id', 'name'])
    output_dir = Path(output_dir)

    standard_messages, standard_members = standardize_data(
        messages, members, author_id_column, content_column)

    standard_messages.to_csv(output_dir / 'raw_messages.csv', index=False)
    standard_members.to_csv(output_dir / 'members.csv', index=False)

    return output_dir

def query(payload_input, HF_TOKEN: str, EOS_TOKEN: str, MODEL_ID: str):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
    messages_list = payload_input.split(EOS_TOKEN)[:-1]
    payload = {
        "inputs": {
            "past_user_inputs": messages_list[:-1],
            "generated_responses": [],
            "text": messages_list[-1],
        }
    }
    payload_dump = json.dumps(payload)
    response = requests.request(
        "POST", API_URL, headers=headers, data=payload_dump)
    return json.loads(response.content.decode("utf-8"))
