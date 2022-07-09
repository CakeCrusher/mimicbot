import configparser
import datetime
from pathlib import Path
import typer
from mimicbot import config, __app_name__
from collections.abc import Callable

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
        delete_folder(app_path)
    return app_path_str


def datetime_str():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")


def callback_config(callback: Callable[[configparser.ConfigParser], configparser.ConfigParser] | None = None, app_path: Path = APP_DIR_PATH / "config.ini") -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    try:
        config.read(str(app_path))
    except:
        raise FileNotFoundError
    if callback:
        config = callback(config)
    with open(str(app_path), "w") as config_file:
        config.write(config_file)
    return config


def session_path(config: configparser.ConfigParser) -> Path:
    GUILD = config.get("discord", "guild")
    DATA_PATH = config.get("general", "data_path")
    SESSION_NAME = config.get("general", "session")
    SESSION_DATA_PATH = Path(DATA_PATH) / Path(GUILD) / Path(SESSION_NAME)
    return SESSION_DATA_PATH
