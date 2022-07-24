import configparser
import datetime
from pathlib import Path
import typer
from mimicbot import config, __app_name__, types
from collections.abc import Callable
import json

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
    if app_path.exists():
        typer.confirm(
            typer.style(
                f"\n[{app_path_str}] config already exists. Do you want to overwrite it?\n", fg=typer.colors.YELLOW),
            False,
            abort=True,
        )
        # rewrite the config file
        config_parser = configparser.ConfigParser()
        with open(str(app_path / "config.ini"), "w") as config_file:
            config_parser.write(config_file)

    return app_path_str


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
            "\nModel to run bot on:\n" + models_string + "Enter numberof model",
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
