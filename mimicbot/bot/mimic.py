import os
import discord
from discord.ext import commands
import requests
import pandas as pd
import json
from DataCleaning import cleanDf
import asyncio
import pdb

MIMICBOT_TOKEN = os.getenv("MIMICBOT_TOKEN")
CHANNEL = os.getenv("CHANNEL")
HF_TOKEN = os.getenv("HF_TOKEN")
AMT_OF_CONTEXT = os.getenv("AMT_OF_CONTEXT")
MODEL_ID = os.getenv("MODEL_ID")
EOS_TOKEN = "<|endoftext|>"
membersDf = None
print("Starting MimicBot")

intents = discord.Intents.default()
intents.members = True
intents.messages = True
bot = commands.Bot(intents=intents, command_prefix="!")


def messagesIntoInput(messages, membersDf):
    messagesDfColumns = ["content", "timestamp", "channel"]
    contextData = [
        [message.content, message.created_at, message.channel.name]
        for message in messages
    ]
    contextDf = pd.DataFrame(columns=messagesDfColumns, data=contextData)
    contextDf = cleanDf(contextDf, membersDf)

    return EOS_TOKEN.join(list(contextDf["content"])) + EOS_TOKEN


def query(payloadInput):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
    messagesList = payloadInput.split(EOS_TOKEN)[:-1]
    payload = {
        "inputs": {
            "past_user_inputs": messagesList[:-1],
            "generated_responses": [],
            "text": messagesList[-1],
        }
    }
    payloadDump = json.dumps(payload)
    response = requests.request(
        "POST", API_URL, headers=headers, data=payloadDump)
    return json.loads(response.content.decode("utf-8"))


@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')


@bot.event
async def on_message(message):
    global membersDf
    guild = discord.utils.get(bot.guilds, name=message.guild.name)
    data = [[member.id, member.name]
            for member in guild.members] + [[bot.user.id, bot.user.name]]
    membersDf = pd.DataFrame(data, columns=["id", "name"])

    if message.author == bot.user:
        return
    # if message is in allowed channel
    channel = message.channel
    # if channel.name == CHANNEL:
    if channel.name == CHANNEL:
        # if bot is mentioned
        if f"<@{bot.user.id}>" in message.content:
            async with channel.typing():
                print(f"{message.author} mentioned me")
                contextMessages = await channel.history(limit=int(AMT_OF_CONTEXT)).flatten()
                payloadText = messagesIntoInput(contextMessages, membersDf)
                queryRes = query(payloadText)
                while "error" in queryRes.keys():
                    # wait for model to load and try again
                    timeToLoad = int(int(queryRes["estimated_time"]) * 1.3)
                    print(
                        f"Waiting for model to load. Will take {timeToLoad}s")
                    await asyncio.sleep(timeToLoad)
                    queryRes = query(payloadText)
                response = queryRes["generated_text"]
                await channel.send(response)

bot.run(MIMICBOT_TOKEN)
