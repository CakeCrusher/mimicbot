import configparser
import datetime
from pathlib import Path
from typing import Tuple
import typer
from mimicbot_cli import config, __app_name__, types
import json
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


def try_session_path(app_path: Path = APP_DIR_PATH):
    try:
        return session_path(callback_config(app_path))
    # except (FileNotFoundError, configparser.NoSectionError):
    except:  # to be safe
        return None


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
        name_as_id = missing_members[0]
        if skip_naming:
            member_name = name_as_id
        else:
            member_name = typer.prompt(
                f'\nMay be the same as id.\nEnter name for member with id ({missing_members[0]})', default=name_as_id)
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
    members["id"] = members["id"].astype(str)
    target_user = current_config('discord', 'target_user')
    if skip_naming and members[members["name"] == target_user].empty:
        target_user_id = None
        while members[members["id"] == target_user_id].empty:
            target_user_id = typer.prompt(
                f"\nEnter id of user to target ({target_user})", default=members["id"].iloc[0])
        members.loc[members["id"] == target_user_id, "name"] = target_user

    return (standard_messages, members)


def save_standardized_data(messages_path: str, members_path: str, output_dir: str, author_id_column: str, content_column: str, forge_pipeline: bool = False) -> Path:
    messages = pd.read_csv(messages_path)
    try:
        members = pd.read_csv(members_path)
    except:
        members = pd.DataFrame(columns=['id', 'name'])
    output_dir = Path(output_dir)

    standard_messages, standard_members = standardize_data(
        messages, members, author_id_column, content_column, forge_pipeline)

    standard_messages.to_csv(output_dir / 'raw_messages.csv', index=False)
    standard_members.to_csv(output_dir / 'members.csv', index=False)

    return output_dir

from transformers import AutoConfig, AutoModelWithLMHead, AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import create_repo, HfApi, get_full_repo_name
from mimicbot_cli import Args, API_KEY_ERROR, CHANGE_VALUE
import os
from urllib.error import HTTPError

def get_model(args: Args, LARGE_LANGUAGE_MODEL: str, MODEL_NAME: str, HUGGINGFACE_API_KEY: str):
    if LARGE_LANGUAGE_MODEL == "microsoft/DialoGPT-small":
        return AutoModelWithLMHead.from_pretrained(
            MODEL_NAME,
            from_tf=False,
            config=config,
            cache_dir=args.cache_dir,
            use_auth_token=HUGGINGFACE_API_KEY,
        )
    else:
        return AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            from_tf=False,
            config=config,
            cache_dir=args.cache_dir,
            use_auth_token=HUGGINGFACE_API_KEY,
        )

def initialize_model(args: Args, HUGGINGFACE_API_KEY: str, LARGE_LANGUAGE_MODEL: str, MODEL_TO: str) -> str: # MODEL_FROM
    print("Initializing model...")
    try:
        config = AutoConfig.from_pretrained(
            LARGE_LANGUAGE_MODEL, cache_dir=args.cache_dir, use_auth_token=HUGGINGFACE_API_KEY)
        
        model = get_model(args, LARGE_LANGUAGE_MODEL, MODEL_TO, HUGGINGFACE_API_KEY)

        MODEL_FROM = MODEL_TO
    except OSError:
        MODEL_FROM = LARGE_LANGUAGE_MODEL
    except ValueError:
        raise ValueError("Huggingface API key is invalid")
        # return (f"https://huggingface.co/{args.save_to}", API_KEY_ERROR)
    if MODEL_FROM != MODEL_TO:
        try:
            link_to_repo = create_repo(
                args.save_to, private=True, token=HUGGINGFACE_API_KEY)
            config = AutoConfig.from_pretrained(
                LARGE_LANGUAGE_MODEL, cache_dir=args.cache_dir, use_auth_token=HUGGINGFACE_API_KEY)
            tokenizer = AutoTokenizer.from_pretrained(
                args.config_name, cache_dir=args.cache_dir, use_auth_token=HUGGINGFACE_API_KEY)
            model = get_model(args, LARGE_LANGUAGE_MODEL, MODEL_FROM, HUGGINGFACE_API_KEY).to(args.device)
            hf_api = HfApi()
            current_dir = os.path.dirname(os.path.abspath(__file__))
            hf_api.upload_file(
                path_or_fileobj=str(Path(current_dir) /
                                    "huggingface/README.md"),
                path_in_repo="README.md",
                repo_id=args.save_to,
                token=HUGGINGFACE_API_KEY)
            # config.push_to_hub(
            #     args.model_path, commit_message="init config", use_auth_token=HUGGINGFACE_API_KEY)
            tokenizer.push_to_hub(
                args.model_path, commit_message="init tokenizer", use_auth_token=HUGGINGFACE_API_KEY)
            model.push_to_hub(
                args.model_path, commit_message="init model", use_auth_token=HUGGINGFACE_API_KEY)
            typer.secho(
                f"\n({datetime.datetime.now().hour}:{datetime.datetime.now().minute}) Huggingface repo initialized at: {link_to_repo}", fg=typer.colors.BLUE)
        except ValueError:
            raise ValueError("Huggingface API key is invalid")
            # return (f"https://huggingface.co/{args.save_to}", API_KEY_ERROR)
        except HTTPError:
            typer.secho(
                f"\n({datetime.datetime.now().hour}:{datetime.datetime.now().minute}) Will update repo at: {'https://huggingface.co/'+MODEL_TO}", fg=typer.colors.BLUE)
        except OSError:
            # cannot delete old model so if model is deleted you need to rename model
            raise OSError("Please rename your model.")
            # return (f"https://huggingface.co/{args.save_to}", CHANGE_VALUE)
    
    return MODEL_FROM

from datasets import load_metric
import numpy as np
rouge_score = load_metric("rouge")
def computeRouge(labels, preds):
    scores = rouge_score.compute(
        predictions=preds, references=labels
    )
    return ({k: np.round(v.mid.fmeasure*100, 4) for k, v in scores.items()}, labels, preds)

def save_to_repo(args, model, message, HUGGINGFACE_API_KEY):
    typer.secho(
        f"\n({datetime.datetime.now().hour}:{datetime.datetime.now().minute}) Uploading model to: https://huggingface.co/{args.save_to}", fg=typer.colors.BLUE)
    model.push_to_hub(
        args.model_path, commit_message=f"model: {message}", use_auth_token=HUGGINGFACE_API_KEY)
    typer.secho(f"\n({datetime.datetime.now().hour}:{datetime.datetime.now().minute}) Uploading finished, view it at: https://huggingface.co/{args.save_to}", fg=typer.colors.BLUE)
