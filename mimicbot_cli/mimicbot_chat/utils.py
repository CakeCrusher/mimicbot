import pandas as pd
import re
import requests
import json
from time import sleep


def clean_df(raw_messages_df: pd.DataFrame, members_df: pd.DataFrame) -> pd.DataFrame:

    # drop all rows with empty content
    raw_messages_df = raw_messages_df[raw_messages_df["content"].notnull()].copy()

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

    # # convert timestamp to uniform datetime
    # raw_messages_df["timestamp"] = pd.to_datetime(
    #     raw_messages_df["timestamp"])

    # # uniformly order by timestamp
    # ordered_df = pd.DataFrame(columns=raw_messages_df.columns)
    # for channel in raw_messages_df["channel"].unique():
    #     channel_messages = raw_messages_df[raw_messages_df["channel"] == channel]
    #     channel_messages = channel_messages.sort_values(by="timestamp")
    #     ordered_df = pd.concat(
    #         [ordered_df, channel_messages], ignore_index=True)
    # raw_messages_df = ordered_df

    # get rid of empty content again
    raw_messages_df["content"] = raw_messages_df["content"].apply(lambda x: x if len(str(x).strip()) else None)
    raw_messages_df = raw_messages_df[raw_messages_df["content"].notnull()].copy()

    raw_messages_df = raw_messages_df.iloc[::-1]

    return raw_messages_df


def messages_into_input(messages: list, members_df, platform: str = 'none', bot=None, eos_token: str = "<|endoftext|>") -> str:
    messages_df_columns = ["content"]
    context_data = [
        [message]
        for message in messages
    ]

    if platform == 'discord':
        # remove first mention of bot from each message
        for idx, context_data_ins in enumerate(context_data):
            message = context_data_ins[0]
            bot_id_decorated = f"<@{bot.user.id}>"
            split_by_bot_id = message.split(bot_id_decorated)
            start_of_message = split_by_bot_id[0]
            rest_of_message = bot_id_decorated.join(split_by_bot_id[1:])
            new_message = (start_of_message + rest_of_message).strip()
            context_data[idx][0] = new_message

    context_df = pd.DataFrame(
        columns=messages_df_columns, data=context_data)
    context_df = clean_df(context_df, members_df)
    return eos_token.join(list(context_df["content"])) + eos_token


def query(payload_input, HF_TOKEN: str, EOS_TOKEN: str, MODEL_ID: str):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
    messages_list = payload_input.split(EOS_TOKEN)[:-1]
    payload = {
        "inputs": {
            "past_user_inputs": messages_list[:-1],
            "generated_responses": [],
            "text": messages_list[-1],
        }
    }
    payload_dump = json.dumps(payload)
    response = requests.request(
        "POST", API_URL, headers=headers, data=payload_dump)
    return json.loads(response.content.decode("utf-8"))


async def respond_to_message(context_messages: list, members_df: pd.DataFrame, respond, log_respond, model_id: str, hf_token: str, EOS_TOKEN: str = "<|endoftext|>", platform: str = 'none', bot=None):
    payload_text = messages_into_input(
        context_messages, members_df, platform=platform, bot=bot)
    # create a string of spaces equal to the context length

    log_respond(payload_text)

    query_res = query(payload_text, hf_token, EOS_TOKEN, model_id)
    attempts = 0

    log_respond(query_res)

    while "error" in query_res.keys() and attempts <= 3:
        # wait for model to load and try again
        if query_res["error"] == "Empty input is invalid":
            break
        time_to_load = int(int(query_res["estimated_time"]) * 1.3)

        log_respond(f"Waiting for model to load. Will take {time_to_load}s")

        sleep(time_to_load)
        query_res = query(payload_text, hf_token, EOS_TOKEN, model_id)

        log_respond(query_res)

        attempts += 1

    if attempts > 3:
        await respond("🤖(failed to load, please try again later)")
        return

    elif "error" in query_res.keys() and query_res["error"] == "Empty input is invalid":
        await respond("...")
        return
    else:
        response: str = query_res["generated_text"]
        if response.strip() == "":
            response = "..."

        if platform == 'discord':
            # allow bot to mention members
            for _idx, member in members_df.iterrows():
                if member["name"] in response:
                    response = response.replace("Hi", f"Hi @{member['name']}")
                    response = response.replace(
                        member["name"], f"<@{member['id']}>")

        await respond(response)
        return
