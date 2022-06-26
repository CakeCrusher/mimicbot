import os
import discord
import pandas as pd
from pathlib import Path
from configparser import ConfigParser
from typing import Tuple

from mimicbot import SUCCESS # pylint: disable=[import-error]



def data_mine(config_path: Path) -> Tuple[Path, int]:
    config = ConfigParser()
    config.read(str(config_path))
    GUILD = config.get("discord", "guild")
    SESSION_NAME = config.get("general", "session")
    app_path = config_path.parent
    GUILD_DATA_PATH = app_path / "data" / GUILD / SESSION_NAME
    print("Starting DataMiner")

    intents = discord.Intents.default()
    # read messaging intent
    intents.messages = True # pylint: disable=[assigning-non-slot]
    intents.members = True # pylint: disable=[assigning-non-slot]
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        CHANNELS_TO_MINE = []  # leave empty to mine all text channels
        guild = discord.utils.get(client.guilds, name=GUILD)
        print(f'{client.user} has connected to Discord!')
        if len(CHANNELS_TO_MINE) == 0:
            CHANNELS_TO_MINE = [
                channel.name for channel in guild.text_channels]

        channels = []

        for channel_name in CHANNELS_TO_MINE:
            channel = discord.utils.get(guild.channels, name=channel_name)
            # breaks if channel is not a text channel
            if not isinstance(channel, discord.TextChannel):
                print(f"{channel_name} is not a text channel")
                raise discord.InvalidArgument
            channels.append(channel)

        # mine data
        messages_columns = ['author_id', 'content', 'timestamp', 'channel']
        messages_data = []
        for channel in channels:
            print("Mining channel: " + channel.name)
            # get messages in channel
            messages = await channel.history(limit=None).flatten()
            messages_for_channel = [
                [message.author.id, message.content,
                    message.created_at, message.channel.name]
                for message in messages
            ]
            messages_data = messages_data + messages_for_channel

        # create a directory for session
        if not GUILD_DATA_PATH.exists():
            os.makedirs(str(GUILD_DATA_PATH))
        # Path make dirs
        # GUILD_DATA_PATH.mkdir(exist_ok=True)

        # write to csv
        messages_df = pd.DataFrame(
            columns=messages_columns, data=messages_data)
        print(messages_df.info())
        messages_df.to_csv(str(GUILD_DATA_PATH / "messages.csv"), index=False)

        # create members reference file
        members_columns = ["id", "name"]
        members_data = [[member.id, member.name] for member in guild.members]
        messages_df = pd.DataFrame(columns=members_columns, data=members_data)
        messages_df.to_csv(str(GUILD_DATA_PATH / "members.csv"), index=False)
        await client.close()

        @client.event
        async def on_error(event, *args, **kwargs):
            print(f'Error on: {event}')
            with open('err.log', 'a') as f:
                f.write(f'Error on: {event}\n')

    client.run(config.get("discord", "api_key"))

    return (GUILD_DATA_PATH, SUCCESS)
