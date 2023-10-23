FROM python:3.8.18-slim-bullseye

COPY ./ ./

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y && \
    pip3 install --upgrade pip && \
    pip3 install -r requirements.txt

ENV BASE_DIR='./'
ENV HYDRA_FULL_ERROR=1

CMD [ "python3", "-m",  "anything_counter.main", "anything_counter/visualizer=dummy"]