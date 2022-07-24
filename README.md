# mimicbot 🤖
## About
Mimicbot is a pipeline that is currently intended for use exclusively in the Discord platform. Mimicbot allows for an effortless yet modular creation of an AI chat bot modeled to imitate a user in the discord channel. It consists of a pipeline that creates the bot from scratch.
## Quickstart
To get started follow the steps below:
1. Clone the repository `git clone https://github.com/CakeCrusher/mimicbot.git`
2. Install the dependencies `pip install -r requirements.txt`. (WARNING: the dependencies will consume a lot of space. If you have an environment with pytorch already installed it is advisable that you use that environment.)
3. Run the command `python -m mimicbot forge`. This command will guide you through the creation of the bot from start to finish.
 
## Commands
Type `python -m mimicbot --help` to see a list of commands. Similarly you can use `python -m mimicbot <command> --help` to see the help for a specific command.
## Deploy
Although technically you could deploy your bot to any server using this repository it is not recommended primarily because the heavy dependencies. Consequently, the [mimicbot-deploy](https://github.com/CakeCrusher/mimicbot-deploy) repository was built for ease of deployment.
Follow the steps listed in its README to deploy your bot.
 
If you are still interested in deploying with this repository you can do so by either running `forge` on the server or passing the configuration files and data files to the appropriate paths on the server and then running `activate`.
## Todo
- [ ] Incorporate github actions to run the pytest tests.
- [ ] Use [DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter) cli as the primary source of mining data to be able to capture dm data in addition to guild data. It also does not require admin access for guilds. The only catch is its security is yet to be determined.
- [ ] Add testing
  - [ ] `activate` command unit tests
  - [ ] `train` command unit tests
  - [ ] End to end tests
 
## Troubleshooting
Errors may occur here are common ones with solutions.
### GPU error
1. Visit [this colab notebook](https://colab.research.google.com/drive/1a196Ev2FJ8U_L__BjTTLFqCXrq9YFhc7?usp=sharing).
2. Copy all file in `/TRAIN_DATA/colab` into the root directory of the notebook.
3. Run the notebook.