import os
from platform import platform
import discord
from discord.ext import commands
import pandas as pd
import json
from mimicbot_cli import (
    config,
    utils,
    data_preprocessing,
    types,
)
import asyncio
import pdb
from pathlib import Path
import typer
import datetime
import mimicbot_cli.mimicbot_chat.utils as chat_utils


def start_mimic(model_save: types.ModelSave):
    config_parser = utils.callback_config()
    MIMICBOT_TOKEN = config_parser.get("discord", "api_key")
    # CHANNEL = os.getenv("CHANNEL")
    HF_TOKEN = config_parser.get("huggingface", "api_key")
    AMT_OF_CONTEXT = int(model_save["context_length"])
    MODEL_ID = "/".join(model_save["url"].split("/")[-2:])
    EOS_TOKEN = "<|endoftext|>"
    # members_df = pd.read_csv(str(Path(model_save["data_path"]) / "members.csv"))
    members_df = None

    typer.secho(
        f"({datetime.datetime.now().hour}:{datetime.datetime.now().minute}) Starting MimicBot.", fg=typer.colors.BLUE)

    intents = discord.Intents.default()
    intents.members = True
    intents.messages = True
    bot = commands.Bot(intents=intents, command_prefix="!")

    @bot.event
    async def on_ready():
        typer.secho(f'\n({datetime.datetime.now().hour}:{datetime.datetime.now().minute}) {bot.user} has been activated.',
                    fg=typer.colors.GREEN)

    @bot.event
    async def on_message(message):
        global members_df
        guild = discord.utils.get(bot.guilds, name=message.guild.name)
        data = [[member.id, member.name]
                for member in guild.members] + [[bot.user.id, bot.user.name]]
        members_df = pd.DataFrame(data, columns=["id", "name"])

        if message.author == bot.user:
            return
        # if message is in allowed channel
        channel = message.channel
        # if channel.name == CHANNEL:
        # if bot is mentioned
        if f"<@{bot.user.id}>" in message.content:
            async with channel.typing():
                typer.secho(f"\n({datetime.datetime.now().hour}:{datetime.datetime.now().minute}) {message.author} mentioned me",
                            fg=typer.colors.BLUE)
                context_messages = await channel.history(limit=int(AMT_OF_CONTEXT)).flatten()
                context_messages = [
                    message.content for message in context_messages]

                def temp_respond(response):
                    typer.secho(
                        f"\n({datetime.datetime.now().hour}:{datetime.datetime.now().minute}) {response}", fg=typer.colors.YELLOW)

                async def respond(response):
                    await channel.send(response)

                await chat_utils.respond_to_message(context_messages, members_df, respond, temp_respond, MODEL_ID, HF_TOKEN, EOS_TOKEN, platform=types.Platform.DISCORD, bot=bot)

    bot.run(MIMICBOT_TOKEN)
