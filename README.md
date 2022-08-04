# mimicbot 🤖

## About
Mimicbot is a pipeline that is currently intended for use exclusively in the Discord platform. Mimicbot enables the effortless yet modular creation of an AI chat bot modeled to imitate a user in the discord channel. For example if Shakespeare is in your channel and the mimicbot is based on him, mimicbot will adjust its responses to how Shakespeare would speak.
It consists of a pipeline that creates the bot from scratch, along with multiple other commands to create the bot in a modular format

https://user-images.githubusercontent.com/37946988/182276645-67ecd78a-b1b0-417f-b0e9-162f0efea30d.mp4

## Important Commands
Type `python -m mimicbot --help` to see a list of commands. Similarly you can use `python -m mimicbot <A_MIMICBOT_COMMAND> --help` to see details on a specific command.
 
## Quickstart
### Verify the following prerequisites are met
- Python 3.x (verify with `python --version` or `python3 --version` or `py --version`)
  - Download from the official [Python website](https://www.python.org/downloads/)
- git-lfs (verify with `git lfs --version`)
  - Install with `git lfs install` or follow the official [git lfs installation guide](https://git-lfs.github.com/).
- pip (verify with `pip --version`)
  - Install with `py get-pip.py` or follow the official [pip installation guide](https://pip.pypa.io/en/stable/installation/).

### Note
If you run into any issues, you dont know how to deal with. Utilize the mimicbot [Dockerfile](https://github.com/CakeCrusher/mimicbot/blob/master/Dockerfile) to [spin up a functional environment](https://github.com/CakeCrusher/mimicbot#spining-up-docker-environement).

### Steps
1. Clone the repository `git clone https://github.com/CakeCrusher/mimicbot.git`
2. Navigate into the cloned directory `cd mimicbot`
3. Install the dependencies `pip install -r requirements.txt`. If you have [CUDA](https://developer.nvidia.com/cuda-downloads) installed use  `pip install -r requirements-gpu.txt` instead. (WARNING: the dependencies will consume a lot of space. If you have an environment with pytorch already installed it is advisable that you use that environment.)
4. Run the command `python -m mimicbot forge`. This command will guide you through the creation of the bot from start to finish.
5. Once the mimicbot is activated on discord, you can interact with it by simply sending it a message as you would to the person it is imitating, with a `@<NAME_OF_MIMICBOT>` mention somewhere in the text. The mimicbot will then reply with a message similar to how the original user would have said it.
 
 
## Deploy
Although technically you could deploy your bot to any server using this repository it is not recommended primarily because of the heavy dependencies. Consequently, the [mimicbot-deploy](https://github.com/CakeCrusher/mimicbot-deploy) repository was built for ease of deployment.
Follow the steps listed in its README to deploy your bot.
 
If you are still interested in deploying with this repository you can do so by either running `forge` on the server or passing the configuration files and data files to the appropriate paths on the server and then running `activate`.


## Todo 
(Feel free to contribute)
- [x] Incorporate github actions to run the pytest tests.
- [ ] Enable bots to mention users in the channel.
- [ ] More error handling.
- [ ] Linux support.
- [ ] Add linting.
  - [ ] github action.
- [ ] Use [DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter) cli as the primary source of mining data to be able to capture dm data in addition to guild data. It also does not require admin access for guilds. The only catch is its security is yet to be determined.
- [ ] Embed training settings into the CLI command
- [ ] More flexible with inputs
  - [ ] Target user input
  - [ ] Guild channel names
- [ ] Make sure the bot learns as it is being used.
- [ ] Create a public server for running through the pipeline and adding deploying the bot. Will work by simply downloading the bot and commanding it to copy a user, then wait a couple minutes and BAM! It's mimicing.
- [ ] Add testing
  - [ ] `activate` command unit tests
  - [ ] `train` command unit tests
  - [ ] End to end tests
 
 
## Mini-guides
Quick guides to clear up the more confusing parts to getting up and running with mimicbot.

### Setting up Discord bot (and retrieving API token)
If you are running `mimicbot forge` on the CLI make sure to complete this guide before submitting the API key.
1. Create a discord developer account and log in.
2. Create a discord application in [applications page of the developer portal](https://discord.com/developers/applications). ![applications page](https://user-images.githubusercontent.com/37946988/180846074-d9f31aa1-2ab4-4389-9b67-95e117731052.png)
3. Add a bot to the application in the bot settings inside your application. ![add bot](https://user-images.githubusercontent.com/37946988/180847396-a88123ae-337d-4716-bf46-d0ba8bc8264b.png)
4. Click on the reset token button which will then prompt you to copy the token, this is the token you will provide to mimicbot. Then activate the "server members intent" and the "message content intent". ![bot settings](https://user-images.githubusercontent.com/37946988/180849646-446334aa-8a41-4d0d-a6b0-f8e60c951182.png)
5. Navigate to OAuth2 > URL Generator then check "bot" as the scope. Then check the "Read Messages/View Channels" general permission and "Send Messages" text permission. Finally, once all of those are checked copy the Generated URL. ![image](https://user-images.githubusercontent.com/37946988/180850821-8816d31f-307f-4a2d-afa1-270becf448e5.png)
6. (You must be signed into your discord account on the browser) Simply navigate to the URL you copied, then select the server (you must have admin privileges) where you want to both: mine data and activate the bot. Click create. ![image](https://user-images.githubusercontent.com/37946988/180852315-b3da1d54-9cbf-4387-a59a-b12ea42706e8.png)
7. Click authorize, click through Captcha, and you're done! Your discord bot is be ready to go! ![image](https://user-images.githubusercontent.com/37946988/180853164-d0645456-2591-4889-b2c2-b1f3dbccf376.png)
8. If you are running `mimicbot forge` make sure to return to the CLI, enter your Discord API token and continue on the CLI.

### Retrieving Huggingface API token
If you are running `mimicbot forge` on the CLI make sure to complete this guide before moving submitting the API key.
1. Create a huggingface account and log in.
2. Navigate to [your settings](https://huggingface.co/settings/profile).
3. Click on the "Access Tokens" tab, then create a "New Token". Then label it, select its "Role" as "write", and press "Generate a token". ![image](https://user-images.githubusercontent.com/37946988/181860877-a2d3f87f-e886-42d4-a4df-4f54eb75707c.png)
4. Copy the "write" token, this is the token you will provide to mimicbot. ![image](https://user-images.githubusercontent.com/37946988/180854416-8370bf8e-0f1a-4175-a492-ed2bd37cd004.png)
5. If you are running `mimicbot forge` make sure to return to the CLI, enter your Huggingface API token and continue on the CLI.

### Spining up Docker Environement
1. Verify you have docker installed and running by using `docker ps`. You may install it from their [official installation page](https://docs.docker.com/get-docker/).
2. Create a new docker image by running `docker build -t mimicbot .`.
3. Run the docker image with `docker run -it --name <ANY_NAME> mimicbot /bin/bash`.
4. You are now ready to start using mimicbot's CLI.
5. You may return to the same container by running `docker start -ai <ANY_NAME>`

 
## Troubleshooting
Errors may occur here are common ones with solutions.

### GPU error
1. Visit [this colab notebook](https://colab.research.google.com/drive/1a196Ev2FJ8U_L__BjTTLFqCXrq9YFhc7?usp=sharing).
2. Copy all file in your `/DATA_PATH/colab` into the root directory of the notebook. (If you don't know what your `DATA_PATH` is, enter the following command in a terminal: `python -m mimicbot config`. Then find the line that with the text `data_path = ...` your `DATA_PATH` is listed there.) ![image](https://user-images.githubusercontent.com/37946988/180862412-5eaf0f84-d5e7-4498-9b58-f1ebaa424eb1.png)
3. Click on the "Edit" tab and then click on the "Notebook settings". Select "GPU" for Hardware accelerator, and finally click "Save". ![image](https://user-images.githubusercontent.com/37946988/180859764-a1e0291a-e81a-4241-8793-1568f4813a1e.png) ![image](https://user-images.githubusercontent.com/37946988/180860154-2e18ee5e-011a-41b6-9bdd-b1b024480622.png)
4. Click on the "Runtime" tab, then click "Run all". ![image](https://user-images.githubusercontent.com/37946988/180862707-4a3b7f59-99da-4ffa-a76d-7f9c8563cf05.png)
5. Wait for the script to finish. You will know it is done with the following indicators: 1. The favicon is yellow, 2. There is a green checkmark next to the cell, and 3. Scroll all the way down to the bottom of the output and you should see a timestamped message saying "Training finished". ![image](https://user-images.githubusercontent.com/37946988/180861730-36662d07-51f7-40ad-86f3-f257ad2cd07b.png)