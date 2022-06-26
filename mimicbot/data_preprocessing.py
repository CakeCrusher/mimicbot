import re
from configparser import ConfigParser
import pandas as pd


def clean_df(raw_messages_data: pd.DataFrame, members_df: pd.DataFrame) -> pd.DataFrame:
    config = ConfigParser()
    GUILD = config.get("discord", "guild")
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

    # create regex that begins with <@ and ends with >
    user_regex = r'<@[0-9]+>'

    def replace_user_token(text):
        span = re.search(user_regex, text).span()
        user_id = text[span[0]+2:span[1]-1]
        user_name = members_df[members_df["id"]
                               == int(user_id)]["name"].iloc[0]
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

    return raw_messages_data


def package_and_save_df():
    GUILD = os.getenv("GUILD")
    SESSION_NAME = os.getenv("SESSION_NAME")
    AUTHOR_ID = os.getenv("AUTHOR_ID")
    AMT_OF_CONTEXT = os.getenv("AMT_OF_CONTEXT")
    cleanedMessages = pd.read_csv(
        f"./data/{GUILD}/{SESSION_NAME}/cleanedMessages.csv")

    cleanedMessages.info()

    from tqdm.auto import tqdm
    uniqueChannels = cleanedMessages["channel"].unique()
    progress_bar = tqdm(range(len(cleanedMessages)-7*len(uniqueChannels)))

    responseAndContext = []
    amtOfContext = int(AMT_OF_CONTEXT)
    for channel in uniqueChannels:
        channelMessages = cleanedMessages[cleanedMessages["channel"] == channel]
        # reset index of channelMessages
        channelMessages = channelMessages.reset_index(drop=True)
        # iterate through each row of channelMessages
        for index, row in channelMessages[7:].iterrows():
            if row["author_id"] == int(AUTHOR_ID):
                rowRaC = []
                for i in range(index, index-amtOfContext-1, -1):
                    rowRaC.append(channelMessages.iloc[i].content)
                responseAndContext.append(rowRaC)
            progress_bar.update(1)
        # break
    responseAndContext

    responseAndContextColumns = ["response"] + \
        ["context" + str(i+1) for i in range(amtOfContext)]
    responseAndContextColumns

    messagesForModel = pd.DataFrame(
        responseAndContext, columns=responseAndContextColumns)
    messagesForModel

    # shuffle and train test split
    shuffledData = messagesForModel.sample(frac=1)
    testPerc = 0.1
    testSize = int(testPerc * len(shuffledData))
    testData = shuffledData[:testSize]
    trainData = shuffledData[testSize:]

    shuffledData.info()

    # save
    # make directory if it does not exist
    if not os.path.exists(f"./data/{GUILD}/{SESSION_NAME}/messagesData"):
        os.makedirs(f"./data/{GUILD}/{SESSION_NAME}/messagesData")
    trainData.to_csv(
        f"./data/{GUILD}/{SESSION_NAME}/messagesData/train.csv", index=False)
    testData.to_csv(
        f"./data/{GUILD}/{SESSION_NAME}/messagesData/test.csv", index=False)

    copyTestData = pd.read_csv(
        f"./data/{GUILD}/{SESSION_NAME}/messagesData/test.csv")
    copyTestData.info()
