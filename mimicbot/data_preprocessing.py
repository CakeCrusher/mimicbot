import re
from configparser import ConfigParser
import pandas as pd
from pathlib import Path
import os
# from tqdm.auto import tqdm
import typer
import numpy as np

from mimicbot import (SUCCESS, DIR_ERROR, USER_NAME_ERROR, config, Args)
from mimicbot.mimicbot_chat.utils import clean_df


def clean_messages(session_path: Path) -> Path:
    if not session_path.exists():
        return (Path(""), DIR_ERROR)

    raw_messages_data = pd.read_csv(session_path / "raw_messages.csv")
    members_df = pd.read_csv(session_path / "members.csv")

    raw_messages_data = clean_df(raw_messages_data, members_df)

    # save data
    raw_messages_data.to_csv(
        session_path / "cleaned_messages.csv", index=False)

    return (session_path / "cleaned_messages.csv", SUCCESS)


def package_data_for_training(cleaned_messages_path: Path, app_path: Path = config.APP_DIR_PATH) -> Path:
    config_parser = ConfigParser()
    config_parser.read(app_path / "config.ini")
    AUTHOR_NAME = config_parser.get("discord", "target_user")
    AMT_OF_CONTEXT = int(config_parser.get("training", "context_length"))
    TEST_PERC = float(config_parser.get("training", "test_perc"))
    CONTEXT_WINDOW = config_parser.get("training", "context_window")
    cleaned_messages = pd.read_csv(cleaned_messages_path)

    members_df = pd.read_csv(cleaned_messages_path.parent / "members.csv")
    try:
        AUTHOR_ID = members_df[members_df["name"] == AUTHOR_NAME]["id"].iloc[0]
    except IndexError:
        return (Path(""), USER_NAME_ERROR)
    # unique_channels = cleaned_messages["channel"].unique()
    # progress_bar = tqdm(range(len(cleaned_messages)-7*len(unique_channels)))

    context_for_base_df = AMT_OF_CONTEXT
    if CONTEXT_WINDOW:
        CONTEXT_WINDOW = int(CONTEXT_WINDOW)
        context_for_base_df = CONTEXT_WINDOW

    response_and_context = []
    # for channel in unique_channels:
    #     channel_messages = cleaned_messages[cleaned_messages["channel"] == channel]
    #     channel_messages = channel_messages.reset_index(drop=True)
    # iterate through each row of channelMessages
    for index, row in cleaned_messages[context_for_base_df:].iterrows():
        if row["author_id"] == AUTHOR_ID:
            row_response_and_context = []
            for i in range(index, index-context_for_base_df-1, -1):
                row_response_and_context.append(
                    cleaned_messages.iloc[i].content)
            response_and_context.append(row_response_and_context)
        # progress_bar.update(1)

    response_and_context_columns = ["response"] + \
        ["context" + str(i+1) for i in range(context_for_base_df)]

    messages_for_model = pd.DataFrame(
        response_and_context, columns=response_and_context_columns
    )

    # shuffle and train test split
    shuffled_data = messages_for_model.sample(frac=1)
    test_size = int(TEST_PERC * len(shuffled_data))
    test_data = shuffled_data[:test_size]
    train_data = shuffled_data[test_size:]

    if CONTEXT_WINDOW:
        train_data = extrapolate_df(train_data, AMT_OF_CONTEXT, CONTEXT_WINDOW)
        test_data = extrapolate_df(test_data, AMT_OF_CONTEXT, CONTEXT_WINDOW)

    # make directory if it does not exist
    training_data_dir = cleaned_messages_path.parent / "training_data"
    if not training_data_dir.exists():
        training_data_dir.mkdir()

    messages_for_model.to_csv(
        str(training_data_dir.parent / "packaged_data.csv"), index=False)

    train_data.to_csv(
        str(training_data_dir / "train.csv"), index=False)

    args = Args()
    # if test data does not have any rows add a row
    if len(test_data) < args.per_gpu_train_batch_size:
        test_data = pd.DataFrame(columns=response_and_context_columns)
        # make test_data a compy of train_data with only the first row
        test_data = train_data.iloc[0:args.per_gpu_train_batch_size]

    test_data.to_csv(
        str(training_data_dir / "test.csv"), index=False)

    return (training_data_dir, SUCCESS)


def get_combos(combos_ref: list, context: list, data_window: int, next_position: int, past_context: list, context_length: int):
    for i in range(data_window-next_position):
        new_context = context[i+next_position]
        following_position = next_position + i + 1
        updated_context = past_context + [new_context]
        if len(updated_context) == context_length:
            combos_ref.append(updated_context)
        else:
            get_combos(combos_ref, context, data_window,
                       following_position, updated_context, context_length)


def extrapolate_df(df: pd.DataFrame, context_length: int, data_window: int) -> pd.DataFrame:
    response_and_context_windowed_columns = ["response"] + \
        ["context" + str(i+1) for i in range(context_length)]
    new_rows = []
    for i, row in df.iterrows():
        context = np.array(row[1:])
        context_combos = []
        get_combos(context_combos, context,
                   data_window, 0, [], context_length)
        # add row[1] to the beggining of each context_combo
        for combo in context_combos:
            combo.insert(0, row[0])
        new_rows = new_rows + context_combos
    return pd.DataFrame(
        new_rows, columns=response_and_context_windowed_columns)
