# 1. Build image:
#       docker build -t mimicbot .
# 2. Run new container (will need to be retrained):
#       docker run -it --name mimicbot mimicbot
# 3. Start old container (select YES on first prompt, your last config should be the default values):
#       docker start -ai mimicbot
# 4. Start old container in bash (good for debugging):
#       docker start mimicbot
#       docker exec -it mimicbot bash
FROM python:3.9

WORKDIR /usr/src/app

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get install git-lfs

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir /root/.config /root/.config/mimicbot_cli

RUN echo "#### NOTE: View Dockerfile to see some recommended commands ####"