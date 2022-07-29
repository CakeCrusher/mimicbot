# mimicbot 🤖
 
 
## About
Mimicbot is a pipeline that is currently intended for use exclusively in the Discord platform. Mimicbot allows for an effortless yet modular creation of an AI chat bot modeled to imitate a user in the discord channel. It consists of a pipeline that creates the bot from scratch.
 
 
## Quickstart
To get started follow the steps below:
1. Clone the repository `git clone https://github.com/CakeCrusher/mimicbot.git`
2. Install the dependencies `pip install -r requirements.txt`. If you have [CUDA](https://developer.nvidia.com/cuda-downloads) installed use  `pip install -r requirements-gpu.txt` instead. (WARNING: the dependencies will consume a lot of space. If you have an environment with pytorch already installed it is advisable that you use that environment.)
3. Run the command `python -m mimicbot forge`. This command will guide you through the creation of the bot from start to finish.
 
 
## Commands
Type `python -m mimicbot --help` to see a list of commands. Similarly you can use `python -m mimicbot <command> --help` to see the help for a specific command.
## Deploy
Although technically you could deploy your bot to any server using this repository it is not recommended primarily because of the heavy dependencies. Consequently, the [mimicbot-deploy](https://github.com/CakeCrusher/mimicbot-deploy) repository was built for ease of deployment.
Follow the steps listed in its README to deploy your bot.
 
If you are still interested in deploying with this repository you can do so by either running `forge` on the server or passing the configuration files and data files to the appropriate paths on the server and then running `activate`.
## Todo
- [x] Incorporate github actions to run the pytest tests.
- [ ] Add linting.
  - [ ] github action.
- [ ] Use [DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter) cli as the primary source of mining data to be able to capture dm data in addition to guild data. It also does not require admin access for guilds. The only catch is its security is yet to be determined.
- [ ] Add testing
  - [ ] `activate` command unit tests
  - [ ] `train` command unit tests
  - [ ] End to end tests
 
 
## Mini-guides
Quick guides to clear up the more confusing parts to getting up and running with mimicbot.
### Setting up Discord bot (and retrieving API token)
1. Create a discord developer account and log in.
2. Create a discord application in [applications page of the developer portal](https://discord.com/developers/applications). ![applications page](https://user-images.githubusercontent.com/37946988/180846074-d9f31aa1-2ab4-4389-9b67-95e117731052.png)
3. Add a bot to the application in the bot settings inside your application. ![add bot](https://user-images.githubusercontent.com/37946988/180847396-a88123ae-337d-4716-bf46-d0ba8bc8264b.png)
4. Click on the reset token button which will then prompt you to copy the token, this is the token you will provide to mimicbot. Then activate the "server members intent" and the "message content intent". ![bot settings](https://user-images.githubusercontent.com/37946988/180849646-446334aa-8a41-4d0d-a6b0-f8e60c951182.png)
5. Navigate to OAuth2 > URL Generator then check "bot" as the scope. Then check the "Read Messages/View Channels" general permission and "Send Messages" text permission. Finally, once all of those are checked copy the Generated URL. ![image](https://user-images.githubusercontent.com/37946988/180850821-8816d31f-307f-4a2d-afa1-270becf448e5.png)
6. (You must be signed into your discord account on the browser) Simply navigate to the URL you copied, then select the server (you must have admin privileges) where you want to both want to mine data and activate the bot. Click create. ![image](https://user-images.githubusercontent.com/37946988/180852315-b3da1d54-9cbf-4387-a59a-b12ea42706e8.png)
7. Click authorize, click through Captcha, and you're done! Your discord bot is be ready to go! ![image](https://user-images.githubusercontent.com/37946988/180853164-d0645456-2591-4889-b2c2-b1f3dbccf376.png)
### Retrieving Huggingface API token
1. Create a huggingface account and log in.
2. Navigate to [your settings](https://huggingface.co/settings/profile).
3. Click on the "Access Tokens" tab, and then copy the "write" token, this is the token you will provide to mimicbot. ![image](https://user-images.githubusercontent.com/37946988/180854416-8370bf8e-0f1a-4175-a492-ed2bd37cd004.png)
 
 
## Troubleshooting
Errors may occur here are common ones with solutions.
### GPU error
1. Visit [this colab notebook](https://colab.research.google.com/drive/1a196Ev2FJ8U_L__BjTTLFqCXrq9YFhc7?usp=sharing).
2. Copy all file in your `/DATA_PATH/colab` into the root directory of the notebook. (If you don't know what your `DATA_PATH` is, enter the following command in a terminal: `python -m mimicbot config`. Then find the line that with the text `data_path = ...` your `DATA_PATH` is listed there.) ![image](https://user-images.githubusercontent.com/37946988/180862412-5eaf0f84-d5e7-4498-9b58-f1ebaa424eb1.png)
3. Click on the "Edit" tab and then click on the "Notebook settings". Select "GPU" for Hardware accelerator, and finally click "Save". ![image](https://user-images.githubusercontent.com/37946988/180859764-a1e0291a-e81a-4241-8793-1568f4813a1e.png) ![image](https://user-images.githubusercontent.com/37946988/180860154-2e18ee5e-011a-41b6-9bdd-b1b024480622.png)
4. Click on the "Runtime" tab, then click "Run all". ![image](https://user-images.githubusercontent.com/37946988/180862707-4a3b7f59-99da-4ffa-a76d-7f9c8563cf05.png)
5. Wait for the script to finish. You will know it is done with the following indicators: 1. The favicon is yellow, 2. There is a green checkmark next to the cell, and 3. Scroll all the way down to the bottom of the output and you should see a timestamped message saying "Training finished". ![image](https://user-images.githubusercontent.com/37946988/180861730-36662d07-51f7-40ad-86f3-f257ad2cd07b.png)