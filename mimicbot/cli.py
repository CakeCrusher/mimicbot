import typer
from mimicbot import (
    ERROR,
    __app_name__,
    SUCCESS,
    DIR_ERROR,
    FILE_ERROR,
    API_KEY_ERROR,
    config,
    utils,
    data_preprocessing,
)
import datetime
from configparser import ConfigParser

from mimicbot.bot.mine import data_mine
from pathlib import Path

app = typer.Typer()
# create a datetime string in the format of YYYY-MM-DD-HH-MM


def datetime_str():

    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")


@app.command()
def init(
    session: str = typer.Option(
        str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")),
        "--session",
        "-s",
        help="Session name for organization of data",
    ),
    app_path: str = typer.Option(
        str(config.APP_DIR_PATH),
        "--app-path",
        "-ap",
        prompt="Path to store mimicbot data",
        help="Path to mimicbot config and user data.",
        callback=utils.app_path_verifier,
    ),
    discord_api_key: str = typer.Option(
        ...,
        "--discord-api-key",
        "-dak",
        prompt="\nGuide to creating discord bot and retrieving the API key: (https://youtube.com/)\nEnter your Discord API key",
        help="API key for the discord bot.",
    ),
    discord_guild: str = typer.Option(
        ...,
        "--discord-guild",
        "-dg",
        prompt="\n(for use in gathering data)\n*you must have admin privilages\nDiscord guild(server) name",
        help="Discord guild(server) name",
    ),
    discord_target_user: str = typer.Option(
        ...,
        "--discord-target-user",
        "-dtu",
        prompt="\n(user to mimic from the discord guild)\nTarget user",
        help="Discord user from guild(server) to mimic.",
    ),
    huggingface_api_key: str = typer.Option(
        ...,
        "--huggingface-api-key",
        "-hak",
        prompt="\nGuide to retrieving huggingface API key: (https://youtube.com/)\nEnter your huggingface API key",
    )
) -> None:
    """Initialize the mimicbot"""
    typer.echo(f"app_path: {app_path}")
    app_path = Path(app_path)
    config.init_app(app_path)
    config.general_config(app_path, session)
    config.discord_config(app_path, discord_api_key,
                          discord_guild, discord_target_user)
    config.huggingface_config(app_path, huggingface_api_key)

    typer.secho("\nSuccessfully initialized mimicbot.", fg=typer.colors.GREEN)


@app.command(name="session")
def set_session(
    session_name: str = typer.Option(
        str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")),
        "--session",
        "-s",
        help="Session name for organization of data",
    ),
    app_path: str = typer.Option(
        str(config.APP_DIR_PATH),
        "--app-path",
        "-ap",
        help="Path to mimicbot data."
    ),
) -> None:
    """Set the session name"""
    app_path: Path = utils.ensure_app_path(Path(app_path))
    config.general_config(app_path, session_name)
    typer.secho(
        f"\nSuccessfully set session name to {session_name}.", fg=typer.colors.GREEN)


@app.command()
def mine(
    app_path: str = typer.Option(
        str(config.APP_DIR_PATH),
        "--app-path",
        "-ap",
        help="Path to mimicbot data."
    )
) -> None:
    """Run the mimicbot"""
    # pass arguments to mimicbot.client.run()
    app_path: Path = utils.ensure_app_path(Path(app_path))

    data_path, error = data_mine(app_path / "config.ini")
    if error:
        typer.secho(f"Error: {ERROR[error]}", fg=typer.colors.RED)
        raise typer.Exit(error)

    typer.secho(
        f"\nSuccessfully mined data. You can find it here [{str(data_path)}]",
        fg=typer.colors.GREEN
    )

    #
