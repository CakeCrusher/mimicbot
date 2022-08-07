import discord
import pandas as pd
from pathlib import Path
from configparser import ConfigParser
from typing import Tuple
import typer

from mimicbot_cli import (  # pylint: disable=[import-error]
    SUCCESS, UNKNOWN_ERROR, API_KEY_ERROR, MISSING_GUILD_ERROR, ABORT, utils
)


def data_mine(config_path: Path, forge_pipeline: bool = False) -> Tuple[Path, int]:
    config = ConfigParser()
    config.read(str(config_path))
    GUILD = config.get("discord", "guild")
    SESSION_NAME = config.get("general", "session")
    DISCORD_API_KEY = config.get("discord", "api_key")
    app_path = config_path.parent
    DATA_PATH = Path(config.get("general", "data_path"))
    GUILD_DATA_PATH = DATA_PATH / GUILD / SESSION_NAME
    # print("Starting DataMiner")

    intents = discord.Intents.default()
    # read messaging intent
    intents.messages = True  # pylint: disable=[assigning-non-slot]
    intents.members = True  # pylint: disable=[assigning-non-slot]
    client = discord.Client(intents=intents)
    client.result = None
    client.finished_mining = False

    @client.event
    async def on_ready():
        CHANNELS_TO_MINE = []  # leave empty to mine all text channels
        guild = discord.utils.get(client.guilds, name=GUILD)
        if not guild:
            typer.secho("Guild not found", fg="red")
            # update global result
            client.result = (Path(""), MISSING_GUILD_ERROR)
            await client.close()
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

        # write to csv
        messages_df = pd.DataFrame(
            columns=messages_columns, data=messages_data)
        print(messages_df.info())
        messages_df.to_csv(
            str(GUILD_DATA_PATH / "raw_messages.csv"), index=False)

        # create members reference file
        members_columns = ["id", "name"]
        members_data = [[member.id, member.name] for member in guild.members]
        members_df = pd.DataFrame(columns=members_columns, data=members_data)

        standard_messages, standard_members = utils.standardize_data(
            messages_df, members_df, "author_id", "content", skip_naming=forge_pipeline)

        standard_messages.to_csv(
            str(GUILD_DATA_PATH / "raw_messages.csv"), index=False)
        standard_members.to_csv(
            str(GUILD_DATA_PATH / "members.csv"), index=False)

        client.finished_mining = True
        await client.close()

        @client.event
        async def on_error(event, *args, **kwargs):
            client.result = (Path(""), UNKNOWN_ERROR)
            await client.close()

    try:
        client.run(DISCORD_API_KEY)
    except discord.LoginFailure:
        client.result = (Path(""), API_KEY_ERROR)

    if client.finished_mining:
        client.result = (GUILD_DATA_PATH, SUCCESS)
    else:
        if not client.result:
            client.result = (Path(""), ABORT)

    return client.result
