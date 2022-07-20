import re
from configparser import ConfigParser
import pandas as pd
from pathlib import Path
import os
# from tqdm.auto import tqdm
import typer
import numpy as np

from mimicbot import (SUCCESS, DIR_ERROR, USER_NAME_ERROR)


def clean_df(raw_messages_df: pd.DataFrame, members_df: pd.DataFrame) -> pd.DataFrame:

    # replace na rows with empty strings
    raw_messages_df["content"] = raw_messages_df["content"].apply(
        lambda x:
        x if pd.notnull(x) else " "
    )

    # replace urls
    url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    def replace_url(text):
        span = re.search(url_regex, text).span()
        return text[:span[0]] + "url" + text[span[1]:]

    def replace_all_url_tokens(text):
        while re.search(url_regex, text) is not None:
            text = replace_url(text)
        return text

    raw_messages_df["content"] = raw_messages_df["content"].apply(
        replace_all_url_tokens)

    # replace discord emojis with their names
    emoji_regex = r'<:.*?:[0-9]+>'

    def replace_emoji(text):
        span = re.search(emoji_regex, text).span()
        emojiName = text[span[0]:span[1]].split(':')[1]
        return text[:span[0]] + emojiName + text[span[1]:]

    def replace_all_emoji_tokens(text):
        while re.search(emoji_regex, text) is not None:
            text = replace_emoji(text)
        return text

    raw_messages_df["content"] = raw_messages_df["content"].apply(
        replace_all_emoji_tokens)
    raw_messages_df.head(3)

    # replace user mentions with their names
    user_regex = r'<@[0-9]+>'

    def replace_user_token(text):
        span = re.search(user_regex, text).span()
        user_id = text[span[0]+2:span[1]-1]
        try:
            user_name = members_df[members_df["id"]
                                   == int(user_id)]["name"].iloc[0]
        except IndexError:
            user_name = "Human"
        return text[:span[0]] + user_name + text[span[1]:]

    def replace_all_user_tokens(text):
        while re.search(user_regex, text) is not None:
            text = replace_user_token(text)
        return text

    raw_messages_df["content"] = raw_messages_df["content"].apply(
        replace_all_user_tokens)

    # get rid of all emoji characters
    raw_messages_df["content"] = raw_messages_df["content"].apply(
        lambda x: re.sub(r'[^\x00-\x7F]+', ' ', x))

    # replace line breaks with spaces
    raw_messages_df["content"] = raw_messages_df["content"].apply(
        lambda x: x.replace("\n", " "))

    # convert timestamp to uniform datetime
    raw_messages_df["timestamp"] = pd.to_datetime(
        raw_messages_df["timestamp"])

    # uniformly order by timestamp
    ordered_df = pd.DataFrame(columns=raw_messages_df.columns)
    for channel in raw_messages_df["channel"].unique():
        channel_messages = raw_messages_df[raw_messages_df["channel"] == channel]
        channel_messages = channel_messages.sort_values(by="timestamp")
        # POTENTIALLY PROBLEMATIC vvv
        ordered_df = pd.concat(
            [ordered_df, channel_messages], ignore_index=True)
    raw_messages_df = ordered_df

    return raw_messages_df


def clean_messages(data_path: Path) -> Path:
    if not data_path.exists():
        return (Path(""), DIR_ERROR)

    raw_messages_data = pd.read_csv(data_path / "raw_messages.csv")
    members_df = pd.read_csv(data_path / "members.csv")

    raw_messages_data = clean_df(raw_messages_data, members_df)

    # save data
    raw_messages_data.to_csv(data_path / "cleaned_messages.csv", index=False)

    return (data_path / "cleaned_messages.csv", SUCCESS)


def package_data_for_training(cleaned_messages_path: Path) -> Path:
    config = ConfigParser()
    config.read(cleaned_messages_path.parent.parent.parent.parent / "config.ini")

    GUILD = config.get("discord", "guild")
    SESSION_NAME = config.get("general", "session")
    AUTHOR_NAME = config.get("discord", "target_user")
    AMT_OF_CONTEXT = int(config.get("training", "context_length"))
    TEST_PERC = float(config.get("training", "test_perc"))
    CONTEXT_WINDOW = config.get("training", "context_window")
    cleaned_messages = pd.read_csv(cleaned_messages_path)

    members_df = pd.read_csv(cleaned_messages_path.parent / "members.csv")
    try:
        AUTHOR_ID = members_df[members_df["name"] == AUTHOR_NAME]["id"].iloc[0]
    except IndexError:
        return (Path(""), USER_NAME_ERROR)
    unique_channels = cleaned_messages["channel"].unique()
    # progress_bar = tqdm(range(len(cleaned_messages)-7*len(unique_channels)))

    context_for_base_df = AMT_OF_CONTEXT
    if CONTEXT_WINDOW:
        CONTEXT_WINDOW = int(CONTEXT_WINDOW)
        context_for_base_df = CONTEXT_WINDOW

    response_and_context = []
    for channel in unique_channels:
        channel_messages = cleaned_messages[cleaned_messages["channel"] == channel]
        channel_messages = channel_messages.reset_index(drop=True)
        # iterate through each row of channelMessages
        for index, row in channel_messages[context_for_base_df:].iterrows():
            if row["author_id"] == int(AUTHOR_ID):
                row_response_and_context = []
                for i in range(index, index-context_for_base_df-1, -1):
                    row_response_and_context.append(
                        channel_messages.iloc[i].content)
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
