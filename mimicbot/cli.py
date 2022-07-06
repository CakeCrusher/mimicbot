import configparser
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
from configparser import ConfigParser

from mimicbot.bot.mine import data_mine
from pathlib import Path

app = typer.Typer()


@app.command()
def init(
    session: str = typer.Option(
        str(utils.datetime_str()),
        "--session",
        "-s",
        help="Session name for organization of data",
    ),
    app_path: str = typer.Option(
        str(config.APP_DIR_PATH),
        "--app-path",
        "-ap",
        help="(WARNING: do not change)\nPath to mimicbot config and user data.",
        callback=utils.app_path_verifier,
    ),
    data_path: str = typer.Option(
        str(config.APP_DIR_PATH / "data"),
        "--data-path",
        "-dp",
        prompt="Path to store data",
        help="Path to mimicbot mined data.",
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
    config.general_config(app_path, data_path, session)
    config.discord_config(app_path, discord_api_key,
                          discord_guild, discord_target_user)
    config.huggingface_config(app_path, huggingface_api_key)

    reccomended_settings = typer.confirm(
        "\nUse reccommended training settings?", default=True)
    if not reccomended_settings:
        context_window = 0
        extrapolate = typer.confirm(
            "\n(the data will be expanded by creating squentially sensitive context combinations based on the context window)\nReccomended if less than 2,000 rows of training data.\nExtrapolate data?", default=True)
        while int(context_window) < 1:
            if extrapolate:
                context_window_text = "(number of previous messages to use for context)\nEnter the size of the context messages window"
            else:
                context_window_text = "Enter the number of context messages to use for training"
            context_window = typer.prompt(
                f"\n*must be greater than 0\n{context_window_text}",
                default=6,
            )
            try:
                context_window = int(context_window)
            except ValueError:
                typer.secho("Invalid input. Please enter a number.",
                            fg=typer.colors.RED)
        context_length: str or int = ""
        if extrapolate:
            context_length = 0
            while int(context_length) < 1 or int(context_length) >= context_window:
                context_length = typer.prompt(
                    f"\n*must be greater than 0 and less than {context_window}\nEnter the amount of context messages",
                    default=2
                )
                try:
                    context_length = int(context_length)
                except ValueError:
                    typer.secho("Invalid input. Please enter a number.",
                                fg=typer.colors.RED)
        test_perc = 0
        while float(test_perc) <= 0 or float(test_perc) >= 1:
            test_perc = typer.prompt(
                "\n*must be a decimal between 0 and 1\nEnter the percentage of data to use for testing",
                default=0.1,
            )
            try:
                test_perc = float(test_perc)
            except ValueError:
                typer.secho("Invalid input. Please enter a number.",
                            fg=typer.colors.RED)
        config.training_config(app_path, str(
            context_window), str(context_length), str(test_perc))
    else:
        config.training_config(app_path, "6", "", "0.1")

    typer.secho("\nSuccessfully initialized mimicbot.", fg=typer.colors.GREEN)


@app.command(name="session")
def set_session(
    session_name: str = typer.Option(
        str(utils.datetime_str()),
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
    config_parser = configparser.ConfigParser()
    try:
        config_parser.read(str(app_path / "config.ini"))
    except:
        pass
    config.general_config(app_path, config_parser.get(
        "general", "data_path"), session_name)
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
    app_path: Path = utils.ensure_app_path(Path(app_path))

    data_path, error = data_mine(app_path / "config.ini")
    if error:
        typer.secho(f"Error: {ERROR[error]}", fg=typer.colors.RED)
        raise typer.Exit(1)

    typer.secho(
        f"\nSuccessfully mined data. You can find it here [{str(data_path)}]",
        fg=typer.colors.GREEN
    )


@app.command(name="preprocess")
def preprocess_data(
    session_path: str = typer.Option(
        None,
        "--session-path",
        "-sp",
        prompt="\nEnter the path to the session data",
        help="Path to mimicbot data."
    ),

) -> None:
    if not Path(session_path).exists():
        typer.secho(
            f"\nError: {ERROR[DIR_ERROR]} does not exist.", fg=typer.colors.RED)
        raise typer.Exit(1)
    session_path = Path(session_path)
    clean_data_path, error = data_preprocessing.clean_messages(session_path)
    if error:
        typer.secho(f"Error: {ERROR[error]}", fg=typer.colors.RED)
        raise typer.Exit(1)

    package_data_for_training, error = data_preprocessing.package_data_for_training(
        clean_data_path)
    if error:
        typer.secho(f"Error: {ERROR[error]}", fg=typer.colors.RED)
        raise typer.Exit(1)

    typer.secho(
        f"\nData is ready for training. You can find it here [{str(package_data_for_training)}]",
        fg=typer.colors.GREEN
    )
