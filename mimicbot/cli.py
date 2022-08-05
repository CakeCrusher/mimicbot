import configparser
from random import random
from types import NoneType
from numpy import spacing
import typer
from mimicbot import (
    ERROR,
    __app_name__,
    SUCCESS,
    DIR_ERROR,
    FILE_ERROR,
    API_KEY_ERROR,
    MISSING_GUILD_ERROR,
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
import mimicbot.mimicbot_chat.utils as chat_utils
from pathlib import Path
import os
import sys
import click
import json
import datetime
from huggingface_hub import get_full_repo_name
import shutil
import pandas as pd
import asyncio


app = typer.Typer()


@app.command(name="init_custom")
def init_custom(
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
        prompt="\nSession name for organization in the data path",
        help="Session name for organization in the data path.",
    ),
    data_path: str = typer.Option(
        utils.current_config("general", "data_path",
                             default=str(config.APP_DIR_PATH / "data")),
        "--data-path",
        "-dp",
        prompt="\nPath to save mimicbot's server data and (AI) model saves",
        help="Path to save mimicbot's server data and (AI) model saves.",
    ),
    target_user: str = typer.Option(
        utils.current_config("discord", "target_user"),
        "--target-user",
        "-tu",
        prompt="""\n(Provide the name, not the id, of the user)\nName of the user to mimic.\nTarget user""",
        help="Name of the user to mimic.",
    ),
    huggingface_api_key: str = typer.Option(
        utils.current_config("huggingface", "api_key"),
        "--huggingface-api-key",
        "-hak",
        prompt="\nGuide to retrieving huggingface API key: https://github.com/CakeCrusher/mimicbot#retrieving-huggingface-api-token \nEnter your huggingface API key",
        help="Huggingface's write key to upload models to your account.",
    ),
    huggingface_model_name: str = typer.Option(
        utils.current_config("huggingface", "model_name",
                             default=f"mimicbot-{str(int(random() * 1000))}"),
        "--huggingface-model-name",
        "-hmn",
        prompt="\nName of your model(AI system which will produce text that mimics the user) to be uploaded and fine-tuned in Huggingface.\nName of the model",
        help="Name of your model(AI system which will produce text that mimics the user) to be uploaded and fine-tuned in Huggingface.",
    ),
    messages_path: str = typer.Option(
        ...,
        "--messages_path",
        "-messp",
        prompt="\nPath to custom messages file",
        help="Path to custom messages file.",
        callback=utils.path_verifier,
    ),
    members_path=typer.Option(
        "",
        "--members_path",
        "-memp",
        prompt="\n(optional)\nPath to members info file",
        help="Path to members info file.",
    ),
    author_id_column: str = typer.Option(
        ...,
        "--author_id_column",
        "-aic",
        prompt="\nName of author column in your custom messages file",
        help="Name of author column in your custom messages file.",

    ),
    content_column: str = typer.Option(
        ...,
        "--content_column",
        "-cc",
        prompt="\nName of content(message) column in your custom messages file",
        help="Name of content(message) column in your custom messages file.",
    ),
    output_dir: str = typer.Option(
        None,
        "--output_dir",
        "-od",
        help="Directory to output standardized files.",
    ),
    forge_pipeline: bool = typer.Option(
        False,
        "--forge-pipeline",
        "-fp",
        help="Is running forge command.",
    ),
) -> None:
    """Initialize and set the config variables for using mimicbot independently of any platform. Requires message data."""

    typer.echo(f"app_path: {app_path}")
    app_path = Path(app_path)
    config.init_app(app_path, Path(data_path))
    config.general_config(app_path, data_path, session)
    config.discord_config(
        app_path,
        utils.current_config("discord", "api_key"),
        "CUSTOM",
        target_user
    )
    config.huggingface_config(
        app_path, huggingface_api_key, huggingface_model_name, utils.current_config("huggingface", "model_saves", "[]"))

    utils.session_path(utils.callback_config()).mkdir(
        parents=True, exist_ok=True)

    if forge_pipeline:
        output_dir = str(utils.session_path(utils.callback_config()))
    elif not output_dir:
        output_dir = typer.prompt(
            "\nDirectory to output standardized files",
            default=str(utils.session_path(utils.callback_config())),
        )

    reccomended_settings = typer.confirm(
        "\nUse reccommended training settings?", default=True)
    if not reccomended_settings:
        context_length = 0
        extrapolate = typer.confirm(
            "\nData will be extrapolated by creating squentially sensitive context combinations based on the context window\nReccomended if less than 2,000 rows of training data.\nExtrapolate data?", default=True)
        while int(context_length) < 1:
            context_length = typer.prompt(
                f"\n*must be greater than 0\Context length is number of messages to use as context for text generation to use for training and using mimicbot.\nContext length",
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
                    f"\n*must be greater than your context length ({context_length})\nContext window is the number of previous messages to use as reference to build/extrapolate context.\n Context window",
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

    if forge_pipeline:
        typer.secho(
            f"\n({datetime.datetime.now().hour}:{datetime.datetime.now().minute}) Initializing step (2/5)", fg=typer.colors.BLUE)
    saved_at = utils.save_standardized_data(
        messages_path, members_path, output_dir, author_id_column, content_column)
    typer.secho(
        f"\n({datetime.datetime.now().hour}:{datetime.datetime.now().minute}) Successfully standardized custom data. You can find it here [{str(saved_at)}].", fg=typer.colors.GREEN)


@app.command(name="init_discord")
def init_discord(
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
        prompt="\nSession name for organization in the data path",
        help="Session name for organization in the data path.",
    ),
    data_path: str = typer.Option(
        utils.current_config("general", "data_path",
                             default=str(config.APP_DIR_PATH / "data")),
        "--data-path",
        "-dp",
        prompt="\nPath to save mimicbot's server data and (AI) model saves",
        help="Path to save mimicbot's server data and (AI) model saves.",
    ),
    discord_api_key: str = typer.Option(
        utils.current_config("discord", "api_key"),
        "--discord-api-key",
        "-dak",
        prompt="\nGuide to setting up the discord bot and retrieving the API key: https://github.com/CakeCrusher/mimicbot#setting-up-discord-bot-and-retrieving-api-token \nEnter your Discord API key",
        help="API key for the discord bot.",
    ),
    discord_guild: str = typer.Option(
        utils.current_config("discord", "guild"),
        "--discord-guild",
        "-dg",
        prompt="\n*you must have admin privilages\nThe guild(server) where data will be gathered and the bot will be running.\nDiscord guild(server) name",
        help="Discord guild(server) name",
    ),
    discord_target_user: str = typer.Option(
        utils.current_config("discord", "target_user"),
        "--discord-target-user",
        "-dtu",
        prompt="""\n(user to mimic from the discord guild)\nProvide only the name without the numbers. For example if the id is "Mimicbot#1234" enter "Mimicbot".\nTarget user""",
        help="Discord user from guild(server) to mimic.",
    ),
    huggingface_api_key: str = typer.Option(
        utils.current_config("huggingface", "api_key"),
        "--huggingface-api-key",
        "-hak",
        prompt="\nGuide to retrieving huggingface API key: https://github.com/CakeCrusher/mimicbot#retrieving-huggingface-api-token \nEnter your huggingface API key",
        help="Huggingface's write key to upload models to your account.",
    ),
    huggingface_model_name: str = typer.Option(
        utils.current_config("huggingface", "model_name",
                             default=f"mimicbot-{str(int(random() * 1000))}"),
        "--huggingface-model-name",
        "-hmn",
        prompt="\nName of your model(AI system which will produce text that mimics the user) to be uploaded and fine-tuned in Huggingface.\nName of the model",
        help="Name of your model(AI system which will produce text that mimics the user) to be uploaded and fine-tuned in Huggingface.",
    ),
    forge_pipeline: bool = typer.Option(
        False,
        "--forge-pipeline",
        "-fp",
        help="Is running forge command.",
    ),
) -> None:
    """Initialize and set the config variables for using mimicbot on Discord."""

    typer.echo(f"app_path: {app_path}")
    app_path = Path(app_path)
    config.init_app(app_path, Path(data_path))
    config.general_config(app_path, data_path, session)
    config.discord_config(app_path, discord_api_key,
                          discord_guild, discord_target_user)
    config.huggingface_config(
        app_path, huggingface_api_key, huggingface_model_name, utils.current_config("huggingface", "model_saves", "[]"))

    utils.session_path(utils.callback_config()).mkdir(
        parents=True, exist_ok=True)

    reccomended_settings = typer.confirm(
        "\nUse reccommended training settings?", default=True)
    if not reccomended_settings:
        context_length = 0
        extrapolate = typer.confirm(
            "\nData will be extrapolated by creating squentially sensitive context combinations based on the context window\nReccomended if less than 2,000 rows of training data.\nExtrapolate data?", default=True)
        while int(context_length) < 1:
            context_length = typer.prompt(
                f"\n*must be greater than 0\Context length is number of messages to use as context for text generation to use for training and using mimicbot.\nContext length",
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
                    f"\n*must be greater than your context length ({context_length})\nContext window is the number of previous messages to use as reference to build/extrapolate context.\n Context window",
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
    """Set individual config variables."""
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
        help="Path to mimicbot config."
    ),
    forge_pipeline: bool = typer.Option(
        False,
        "--forge-pipeline",
        "-fp",
        help="Is running forge command.",
    ),
) -> None:
    """Scrape all the message data from the discord server."""
    typer.secho(
        f"\n({datetime.datetime.now().hour}:{datetime.datetime.now().minute}) Begginging to mine data.", fg=typer.colors.BLUE)
    app_path: Path = utils.ensure_app_path(Path(app_path))

    data_path, error = data_mine(app_path / "config.ini", forge_pipeline=forge_pipeline)
    if error:
        if error == MISSING_GUILD_ERROR:
            typer.secho(
                f"Error: Please make sure your bot is connected to the server", fg=typer.colors.RED)
        else:
            typer.secho(f"Error: {ERROR[error]}", fg=typer.colors.RED)
        raise typer.Exit(1)

    typer.secho(
        f"""\n({datetime.datetime.now().hour}:{datetime.datetime.now().minute}) Successfully mined data. You can find it here [{str(data_path)}]. (Ignore the "RuntimeError: Event loop is closed" error show below)""",
        fg=typer.colors.GREEN
    )


@app.command(name="preprocess")
def preprocess_data(
    app_path: str = typer.Option(
        str(config.APP_DIR_PATH),
        "--app-path",
        "-ap",
        help="Path to mimicbot config."
    ),
    session_path: str = typer.Option(
        utils.try_session_path(),
        "--session-path",
        "-sp",
        prompt="Enter the path to the session data",
        help="Path to session data.",
        callback=utils.path_verifier
    ),
    forge_pipeline: bool = typer.Option(
        False,
        "--forge-pipeline",
        "-fp",
        help="Is running forge command.",
    ),
) -> None:
    """Preprocess the data such that it is in a standardized format. Then prepares the data for training."""

    session_path = Path(session_path)
    clean_data_path, error = data_preprocessing.clean_messages(session_path)
    if error:
        typer.secho(f"Error: {ERROR[error]}", fg=typer.colors.RED)
        raise typer.Exit(1)

    packaged_data_for_training, error = data_preprocessing.package_data_for_training(
        clean_data_path, Path(app_path))
    if error:
        typer.secho(f"Error: {ERROR[error]}", fg=typer.colors.RED)
        raise typer.Exit(1)

    typer.secho(
        f"\n({datetime.datetime.now().hour}:{datetime.datetime.now().minute}) Data is ready for training. You can find it here [{str(packaged_data_for_training)}]",
        fg=typer.colors.GREEN
    )


@app.command(name="train")
def train_model(
    app_path: str = typer.Option(
        str(config.APP_DIR_PATH),
        "--app-path",
        "-ap",
        help="Path to mimicbot config."
    ),
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
    """Trains the model to immitate the user identified in the config."""

    app_path = Path(app_path)
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
                f"Your GPU ran out of memory.\nAmong the many ways of going about solving this, here is a quick one: https://github.com/CakeCrusher/mimicbot#gpu-error .", fg=typer.colors.RED)

            config_parser = utils.callback_config()
            colab_path = Path(utils.session_path(config_parser)) / "colab"
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
            utils.add_model_save(app_path, model_save)

        typer.secho(f"Error: {ERROR[error]}", fg=typer.colors.RED)
        if forge_pipeline:
            # create a confirmation prompt to delete the session data
            if typer.confirm(
                f"Colab notebook successfully finished training (continue forge)?", default=True
            ):
                raise typer.Exit(0)
            else:
                raise typer.Exit(1)
        else:
            raise typer.Exit(1)
    config_parser = utils.callback_config()
    context_length = int(config_parser.get("training", "context_length"))
    model_save = {
        "url": res,
        "context_length": context_length,
        "data_path": str(session_path),
    }
    utils.add_model_save(app_path, model_save)

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
    """Activates the discord bot with a trained mimicbot model."""
    config_parser = utils.callback_config()
    model_saves: list[types.ModelSave] = json.loads(
        config_parser.get("huggingface", "model_saves"))
    if not model_idx:
        model_idx = 0
        if not forge_pipeline:
            model_idx = utils.prompt_model_save()
    else:
        model_idx = int(model_idx)
    model_save = model_saves[model_idx]
    start_mimic(model_save)


@app.command(name="chat")
def chat(
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
    """Activates the discord bot with a trained mimicbot model."""

    typer.secho("Chat ready, start chatting with the bot!",
                fg=typer.colors.GREEN)

    config_parser = utils.callback_config()

    model_saves: list[types.ModelSave] = json.loads(
        config_parser.get("huggingface", "model_saves"))
    if not model_idx:
        model_idx = 0
        if not forge_pipeline:
            model_idx = utils.prompt_model_save()
    else:
        model_idx = int(model_idx)

    model_save = model_saves[model_idx]

    HF_TOKEN = utils.current_config("huggingface", "api_key")
    AMT_OF_CONTEXT = int(model_save["context_length"])
    EOS_TOKEN = "<|endoftext|>"
    MODEL_ID = "/".join(model_save["url"].split("/")[-2:])
    MODEL_NAME = model_save["url"].split("/")[-1:][0]
    members_df = pd.read_csv(
        str(Path(model_save["data_path"]) / "members.csv"))

    chat_history = [" " for i in range(AMT_OF_CONTEXT)]

    spacing = " ".join(["" for i in range(250)])

    async def respond(response):
        text = f"\r({MODEL_NAME}): {response}{spacing}"[:250]
        text = typer.style(
            text, fg=typer.colors.BLUE)
        sys.stdout.write(text)
        sys.stdout.flush()
        # push response to the start of the chat history
        chat_history.insert(0, response)

    def temp_respond(response):
        text = f"\r({MODEL_NAME}) WRITING: {response}{spacing}"[:100]
        text = typer.style(
            text, fg=typer.colors.YELLOW)
        sys.stdout.write(text)
        sys.stdout.flush()

    speaker_bot = False
    while True:
        if speaker_bot:
            context_messages = chat_history[:AMT_OF_CONTEXT]
            loop = asyncio.get_event_loop()
            loop.run_until_complete(
                chat_utils.respond_to_message(context_messages, members_df, respond, temp_respond, MODEL_ID, HF_TOKEN, EOS_TOKEN))
            speaker_bot = False
        else:
            response = typer.prompt("\n(You)")
            typer.echo("")
            chat_history.insert(0, response)
            speaker_bot = True


@app.command(name="forge")
def forge(
    discord_bot: bool = typer.Option(
        True,
        "--discord-bot",
        "-db",
        prompt="\nCreating a discord bot or an independently ran bot (requires custom data).\nForge a discord bot?",
        help="\nCreating a discord bot or an independently ran bot (requires custom data).",
    )
):
    """All encompassing command to produce a bot from scratch. It runs the following commands in order: init, mine, preprocess, train, activate"""
    typer.secho(
        f"\n({datetime.datetime.now().hour}:{datetime.datetime.now().minute}) Initializing step (1/5)", fg=typer.colors.BLUE)
    if discord_bot:
        res = os.system("python -m mimicbot init_discord --forge-pipeline")
    else:
        res = os.system("python -m mimicbot init_custom --forge-pipeline")

    session_path = utils.session_path(utils.callback_config())

    if res != 0:
        raise typer.Exit(1)
    if discord_bot:
        typer.secho(
            f"\n({datetime.datetime.now().hour}:{datetime.datetime.now().minute}) Initializing step (2/5)", fg=typer.colors.BLUE)
        res = os.system("python -m mimicbot mine --forge-pipeline")
        if res != 0:
            raise typer.Exit(1)
    typer.secho(
        f"\n({datetime.datetime.now().hour}:{datetime.datetime.now().minute}) Initializing step (3/5)", fg=typer.colors.BLUE)
    res = os.system(
        f"python -m mimicbot preprocess --forge-pipeline -sp \"{str(session_path)}\"")
    if res != 0:
        raise typer.Exit(1)
    typer.secho(
        f"\n({datetime.datetime.now().hour}:{datetime.datetime.now().minute}) Initializing step (4/5)", fg=typer.colors.BLUE)
    res = os.system("python -m mimicbot train --forge-pipeline")
    if res != 0:
        raise typer.Exit(1)
    typer.secho(
        f"\n({datetime.datetime.now().hour}:{datetime.datetime.now().minute}) Initializing final step (5/5)", fg=typer.colors.BLUE)
    if discord_bot:
        res = os.system("python -m mimicbot activate --forge-pipeline")
        if res != 0:
            raise typer.Exit(1)
    else:
        res = os.system("python -m mimicbot chat --forge-pipeline")
        if res != 0:
            raise typer.Exit(1)


@app.command(name="poduction_env")
def generate_poduction_env():
    """Generated an environement file for production."""
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

    deploy_path = Path(utils.session_path(config_parser)) / "deploy"
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


@app.command(name="config")
def get_config(
    app_path: str = typer.Option(
        str(utils.APP_DIR_PATH),
        "--app-path",
        "-ap",
        help="Path to the app directory.",
    ),
):
    """Print the current config."""
    config_path = Path(app_path) / "config.ini"
    # print the content inside config_path
    with open(str(config_path), "r") as f:
        print(f.read())
    typer.secho(
        f"\n({datetime.datetime.now().hour}:{datetime.datetime.now().minute}) Successfully printed the config above.",
        fg=typer.colors.GREEN)


@app.command(name="standardize_data")
def standardize_data(
    messages_path: str = typer.Option(
        ...,
        "--messages_path",
        "-messp",
        prompt="\nPath to custom messages file",
        help="Path to custom messages file.",

    ),
    members_path=typer.Option(
        "",
        "--members_path",
        "-memp",
        prompt="\n(optional)\nPath to members info file",
        help="Path to members info file.",

    ),
    output_dir: str = typer.Option(
        utils.try_session_path(),
        "--output_dir",
        "-od",
        prompt="\nDirectory to output standardized files",
        help="Directory to output standardized files.",

    ),
    author_id_column: str = typer.Option(
        ...,
        "--author_id_column",
        "-aic",
        prompt="\nName of author column in your custom messages file",
        help="Name of author column in your custom messages file.",

    ),
    content_column: str = typer.Option(
        ...,
        "--content_column",
        "-cc",
        prompt="\nName of content(message) column in your custom messages file",
        help="Name of content(message) column in your custom messages file.",
    ),
):
    utils.save_standardized_data(
        messages_path, members_path, output_dir, author_id_column, content_column)
    typer.secho(
        f"\n({datetime.datetime.now().hour}:{datetime.datetime.now().minute}) Successfully standardized data at output directory ({str(output_dir)}).",
        fg=typer.colors.GREEN
    )
