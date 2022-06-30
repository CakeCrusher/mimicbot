import re
from configparser import ConfigParser
import pandas as pd
from pathlib import Path
import os
# from tqdm.auto import tqdm
import typer

from mimicbot import (SUCCESS, DIR_ERROR, USER_NAME_ERROR)


def clean_messages(data_path: Path) -> Path:
    raw_messages_data = pd.read_csv(data_path / "raw_messages.csv")
    members_df = pd.read_csv(data_path / "members.csv")

    if not data_path.exists():
        return (Path(""), DIR_ERROR)

    # replace na rows with empty strings
    raw_messages_data["content"] = raw_messages_data["content"].apply(
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

    raw_messages_data["content"] = raw_messages_data["content"].apply(
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

    raw_messages_data["content"] = raw_messages_data["content"].apply(
        replace_all_emoji_tokens)
    raw_messages_data.head(3)

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

    raw_messages_data["content"] = raw_messages_data["content"].apply(
        replace_all_user_tokens)

    # get rid of all emoji characters
    raw_messages_data["content"] = raw_messages_data["content"].apply(
        lambda x: re.sub(r'[^\x00-\x7F]+', ' ', x))

    # replace line breaks with spaces
    raw_messages_data["content"] = raw_messages_data["content"].apply(
        lambda x: x.replace("\n", " "))

    # convert timestamp to uniform datetime
    raw_messages_data["timestamp"] = pd.to_datetime(
        raw_messages_data["timestamp"])

    # uniformly order by timestamp
    ordered_df = pd.DataFrame(columns=raw_messages_data.columns)
    for channel in raw_messages_data["channel"].unique():
        channel_messages = raw_messages_data[raw_messages_data["channel"] == channel]
        channel_messages = channel_messages.sort_values(by="timestamp")
        # POTENTIALLY PROBLEMATIC vvv
        ordered_df = pd.concat(
            [ordered_df, channel_messages], ignore_index=True)
    raw_messages_data = ordered_df

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
    cleaned_messages = pd.read_csv(cleaned_messages_path)

    members_df = pd.read_csv(cleaned_messages_path.parent / "members.csv")
    try:
        AUTHOR_ID = members_df[members_df["name"] == AUTHOR_NAME]["id"].iloc[0]
    except IndexError:
        return (Path(""), USER_NAME_ERROR)
    unique_channels = cleaned_messages["channel"].unique()
    # progress_bar = tqdm(range(len(cleaned_messages)-7*len(unique_channels)))

    response_and_context = []
    for channel in unique_channels:
        channel_messages = cleaned_messages[cleaned_messages["channel"] == channel]
        channel_messages = channel_messages.reset_index(drop=True)
        # iterate through each row of channelMessages
        for index, row in channel_messages[AMT_OF_CONTEXT + 1:].iterrows():
            if row["author_id"] == int(AUTHOR_ID):
                row_response_and_context = []
                for i in range(index, index-AMT_OF_CONTEXT-1, -1):
                    row_response_and_context.append(
                        channel_messages.iloc[i].content)
                response_and_context.append(row_response_and_context)
            # progress_bar.update(1)

    response_and_context_columns = ["response"] + \
        ["context" + str(i+1) for i in range(AMT_OF_CONTEXT)]

    messages_for_model = pd.DataFrame(
        response_and_context, columns=response_and_context_columns
    )

    # shuffle and train test split
    shuffled_data = messages_for_model.sample(frac=1)
    test_size = int(TEST_PERC * len(shuffled_data))
    test_data = shuffled_data[:test_size]
    train_data = shuffled_data[test_size:]

    # make directory if it does not exist
    training_data_dir = cleaned_messages_path.parent / "training_data"
    if not training_data_dir.exists():
        training_data_dir.mkdir()

    train_data.to_csv(
        str(training_data_dir / "train.csv"), index=False)
    test_data.to_csv(
        str(training_data_dir / "test.csv"), index=False)

    return (training_data_dir, SUCCESS)
