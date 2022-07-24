import configparser
from random import random
import typer
from mimicbot import (
    ERROR,
    __app_name__,
    SUCCESS,
    DIR_ERROR,
    FILE_ERROR,
    API_KEY_ERROR,
    GPU_ERROR,
    CHANGE_VALUE,
    config,
    utils,
    data_preprocessing,
    train,
    types,
)
from configparser import ConfigParser

from mimicbot.bot.mine import data_mine
from mimicbot.bot.mimic import start_mimic
from pathlib import Path
import os
import click
import json
import datetime
from huggingface_hub import get_full_repo_name
import shutil


app = typer.Typer()


@app.command()
def init(
    app_path: str = typer.Option(
        str(config.APP_DIR_PATH),
        "--app-path",
        "-ap",
        help="(WARNING: do not change)\nPath to mimicbot config and user data.",
        callback=utils.app_path_verifier,
    ),
    session: str = typer.Option(
        utils.current_config("general", "session",
                             default=str(utils.datetime_str())),
        "--session",
        "-s",
        prompt="\nSession name",
        help="Session name for organization of data",
    ),
    data_path: str = typer.Option(
        utils.current_config("general", "data_path",
                             default=str(config.APP_DIR_PATH / "data")),
        "--data-path",
        "-dp",
        prompt="\nPath to store data",
        help="Path to mimicbot mined data.",
    ),
    discord_api_key: str = typer.Option(
        utils.current_config("discord", "api_key"),
        "--discord-api-key",
        "-dak",
        prompt="\nGuide to creating discord bot and retrieving the API key: (https://youtube.com/)\nEnter your Discord API key",
        help="API key for the discord bot.",
    ),
    discord_guild: str = typer.Option(
        utils.current_config("discord", "guild"),
        "--discord-guild",
        "-dg",
        prompt="\n(for use in gathering data)\n*you must have admin privilages\nDiscord guild(server) name",
        help="Discord guild(server) name",
    ),
    discord_target_user: str = typer.Option(
        utils.current_config("discord", "target_user"),
        "--discord-target-user",
        "-dtu",
        prompt="\n(user to mimic from the discord guild)\nTarget user",
        help="Discord user from guild(server) to mimic.",
    ),
    huggingface_api_key: str = typer.Option(
        utils.current_config("huggingface", "api_key"),
        "--huggingface-api-key",
        "-hak",
        prompt="\nGuide to retrieving huggingface API key: (https://youtube.com/)\nEnter your huggingface API key",
        help="Huggingface's write key to upload models to your account.",
    ),
    huggingface_model_name: str = typer.Option(
        utils.current_config("huggingface", "model_name",
                             default=f"mimicbot-{str(int(random() * 1000))}"),
        "--huggingface-model-name",
        "-hmn",
        prompt="\nEnter the name of the model",
        help="Name of the model to be uploaded or be fine-tuned huggingface.",
    )
) -> None:
    """Initialize the mimicbot"""

    typer.echo(f"app_path: {app_path}")
    app_path = Path(app_path)
    config.init_app(app_path)
    config.general_config(app_path, data_path, session)
    config.discord_config(app_path, discord_api_key,
                          discord_guild, discord_target_user)
    config.huggingface_config(
        app_path, huggingface_api_key, huggingface_model_name, "[]")

    reccomended_settings = typer.confirm(
        "\nUse reccommended training settings?", default=True)
    if not reccomended_settings:
        context_length = 0
        extrapolate = typer.confirm(
            "\n(the data will be expanded by creating squentially sensitive context combinations based on the context window)\nReccomended if less than 2,000 rows of training data.\nExtrapolate data?", default=True)
        while int(context_length) < 1:
            # if extrapolate:
            #     context_window_text = "(number of previous messages to use for context)\nEnter the size of the context messages window"
            context_window_text = "Enter the context length (number of context messages) to use for training"
            context_length = typer.prompt(
                f"\n*must be greater than 0\n{context_window_text}",
                default=2,
            )
            try:
                context_length = int(context_length)
            except ValueError:
                typer.secho("Invalid input. Please enter a number.",
                            fg=typer.colors.RED)
        context_window: str or int = ""
        if extrapolate:
            context_window = 0
            while int(context_window) <= context_length:
                context_window = typer.prompt(
                    f"\n*must be greater than your context length ({context_length})\nEnter the context window (number of previous messages to use as reference to build context)",
                    default=6
                )
                try:
                    context_window = int(context_window)
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
        config.training_config(app_path, "", "2", "0.1")

    typer.secho(
        f"\n({datetime.datetime.now().hour}:{datetime.datetime.now().minute}) Successfully initialized mimicbot.", fg=typer.colors.GREEN)


@app.command(name="set")
def set_config(
    session_name: str = typer.Option(
        None,
        "--session",
        "-s",
        help="Session name for organization of data",
    ),
    model_name: str = typer.Option(
        None,
        "--model_name",
        "-mn",
        help="Name of the model to be uploaded or be fine-tuned huggingface.",
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
    if session_name:
        config.general_config(app_path, config_parser.get(
            "general", "data_path"), session_name)
    if model_name:
        config.huggingface_config(app_path, config_parser.get(
            "huggingface", "api_key"), model_name, config_parser.get("huggingface", "model_saves"))
    typer.secho(
        f"\nSuccessfully set value.", fg=typer.colors.GREEN)


@app.command()
def mine(
    app_path: str = typer.Option(
        str(config.APP_DIR_PATH),
        "--app-path",
        "-ap",
        help="Path to mimicbot data."
    ),
    forge_pipeline: bool = typer.Option(
        False,
        "--forge-pipeline",
        "-fp",
        help="Is running forge command.",
    ),
) -> None:
    """Run the mimicbot"""
    typer.secho(
        f"\n({datetime.datetime.now().hour}:{datetime.datetime.now().minute}) Begginging to mine data.", fg=typer.colors.BLUE)
    app_path: Path = utils.ensure_app_path(Path(app_path))

    data_path, error = data_mine(app_path / "config.ini")
    if error:
        typer.secho(f"Error: {ERROR[error]}", fg=typer.colors.RED)
        raise typer.Exit(1)

    typer.secho(
        f"\n({datetime.datetime.now().hour}:{datetime.datetime.now().minute}) Successfully mined data. You can find it here [{str(data_path)}]",
        fg=typer.colors.GREEN
    )


@app.command(name="preprocess")
def preprocess_data(
    session_path: str = typer.Option(
        None,
        "--session-path",
        "-sp",
        help="Path to session data."
    ),
    forge_pipeline: bool = typer.Option(
        False,
        "--forge-pipeline",
        "-fp",
        help="Is running forge command.",
    ),
) -> None:

    while not session_path or not Path(session_path).exists():
        config_parser = utils.callback_config()
        session_path = utils.session_path(config_parser)
        if not forge_pipeline:
            session_path = typer.prompt(
                f"\nEnter the path to the session data", default=str(session_path)
            )

    session_path = Path(session_path)
    clean_data_path, error = data_preprocessing.clean_messages(session_path)
    if error:
        typer.secho(f"Error: {ERROR[error]}", fg=typer.colors.RED)
        raise typer.Exit(1)

    packaged_data_for_training, error = data_preprocessing.package_data_for_training(
        clean_data_path)
    if error:
        typer.secho(f"Error: {ERROR[error]}", fg=typer.colors.RED)
        raise typer.Exit(1)

    typer.secho(
        f"\n({datetime.datetime.now().hour}:{datetime.datetime.now().minute}) Data is ready for training. You can find it here [{str(packaged_data_for_training)}]",
        fg=typer.colors.GREEN
    )


@app.command(name="train")
def train_model(
    session_path: str = typer.Option(
        None,
        "--session-path",
        "-sp",
        help="Path to session data."
    ),
    forge_pipeline: bool = typer.Option(
        False,
        "--forge-pipeline",
        "-fp",
        help="Is running forge command.",
    ),
):

    while not session_path or not Path(session_path).exists():
        config_parser = utils.callback_config()
        session_path = utils.session_path(config_parser)
        if not forge_pipeline:
            session_path = typer.prompt(
                f"\nEnter the path to the session data", default=str(session_path)
            )
    session_path = Path(session_path)

    typer.secho(
        f"\n({datetime.datetime.now().hour}:{datetime.datetime.now().minute}) Training model. This may take a while.", fg=typer.colors.YELLOW
    )

    res, error = train.train(session_path)

    if error:
        # create a switch statement

        if (error == CHANGE_VALUE):
            typer.secho(
                f"Error: Please change model name.\nYou may do so with the following command < python -m mimicbot set -mn MODEL_NAME_HERE >", fg=typer.colors.RED)
            raise typer.Exit(1)
        elif (error == GPU_ERROR):
            typer.secho(
                f"Your GPU ran out of memory.\nAmong the many ways of going about solving this, here is a quick one: .", fg=typer.colors.RED)

            config_parser = utils.callback_config()
            colab_path = Path(utils.current_config(
                "general", "data_path")) / "colab"
            colab_path.mkdir(exist_ok=True)

            # write .env in /DATA_PATH/colab/
            HUGGINGFACE_API_KEY = utils.current_config(
                "huggingface", "api_key")
            MODEL_NAME = utils.current_config("huggingface", "model_name")
            with open(str(colab_path / ".env"), "w") as f:
                f.write(
                    f"HUGGINGFACE_API_KEY={HUGGINGFACE_API_KEY}\nMODEL_NAME={MODEL_NAME}")

            # copy training files and paste into /colab/
            DATA_PATH = utils.current_config("general", "data_path")
            GUILD = utils.current_config("discord", "guild")
            SESSION = utils.current_config("general", "session")
            path_to_training_data = Path(
                DATA_PATH) / GUILD / SESSION / "training_data"
            print(path_to_training_data)
            print(colab_path)
            shutil.copytree(str(path_to_training_data),
                            str(colab_path), dirs_exist_ok=True)

            # copy huggingface\README.md into /colab/
            # get current directory
            path_to_readme = Path(__file__).parent / \
                "huggingface" / "README.md"
            shutil.copy(str(path_to_readme), str(colab_path))

            # create a model_save for future use
            context_length = int(config_parser.get(
                "training", "context_length"))
            model_save = {
                "url": f"https://huggingface.co/{get_full_repo_name(MODEL_NAME)}",
                "context_length": context_length,
                "data_path": str(session_path),
            }
            utils.add_model_save(session_path.parent.parent.parent, model_save)

            raise typer.Exit(1)
        typer.secho(f"Error: {ERROR[error]}", fg=typer.colors.RED)
        raise typer.Exit(1)
    config_parser = utils.callback_config()
    context_length = int(config_parser.get("training", "context_length"))
    model_save = {
        "url": res,
        "context_length": context_length,
        "data_path": str(session_path),
    }
    utils.add_model_save(session_path.parent.parent.parent, model_save)

    typer.secho(
        f"\n({datetime.datetime.now().hour}:{datetime.datetime.now().minute}) Successfully trained and saved the model. You can find it here [{str(res)}]",
        fg=typer.colors.GREEN
    )


@app.command(name="activate")
def activate_bot(
    model_idx=typer.Option(
        None,
        "--model-idx",
        "-mi",
        help="Index of the model to be activated."
    ),
    forge_pipeline: bool = typer.Option(
        False,
        "--forge-pipeline",
        "-fp",
        help="Is running forge command.",
    ),
):
    # create a multiple choice question
    # ask the user to select the bot to activate
    config_parser = utils.callback_config()
    model_saves: list[types.ModelSave] = json.loads(
        config_parser.get("huggingface", "model_saves"))
    model_idx = utils.prompt_model_save()
    model_save = model_saves[model_idx]
    print(model_save)
    # start_mimic(model_save)


@app.command(name="forge")
def forge(
):
    res = os.system("python -m mimicbot init")
    if res != 0:
        raise typer.Exit(1)
    res = os.system("python -m mimicbot mine --forge-pipeline")
    if res != 0:
        raise typer.Exit(1)
    res = os.system("python -m mimicbot preprocess --forge-pipeline")
    if res != 0:
        raise typer.Exit(1)
    res = os.system("python -m mimicbot train --forge-pipeline")
    if res != 0:
        raise typer.Exit(1)
    res = os.system("python -m mimicbot activate --forge-pipeline")
    if res != 0:
        raise typer.Exit(1)


@app.command(name="env")
def generate_env():
    config_parser = utils.callback_config()
    model_saves: list[types.ModelSave] = json.loads(
        config_parser.get("huggingface", "model_saves"))
    model_idx = utils.prompt_model_save()
    model_save: types.ModelSave = model_saves[model_idx]

    DISCORD_API_KEY = utils.current_config("discord", "api_key")
    HUGGINGFACE_API_KEY = utils.current_config("huggingface", "api_key")
    CONTEXT_LENGTH = model_save["context_length"]
    MODEL_ID = "/".join(model_save["url"].split("/")[-2:])

    # open a file under /etc/environment
    deploy_path = Path(utils.current_config("general", "data_path")) / "deploy"
    deploy_path.mkdir(exist_ok=True)
    with open(str(deploy_path / ".env"), "w") as f:
        f.write(
            f"DISCORD_API_KEY={DISCORD_API_KEY}\nHUGGINGFACE_API_KEY={HUGGINGFACE_API_KEY}\nCONTEXT_LENGTH={CONTEXT_LENGTH}\nMODEL_ID={MODEL_ID}")

    # get current directory with Path
    env_path = deploy_path / ".env"

    typer.secho(
        f"\n({datetime.datetime.now().hour}:{datetime.datetime.now().minute}) Successfully generated .env file it is located in this directory [{str(env_path)}].",
        fg=typer.colors.GREEN
    )
