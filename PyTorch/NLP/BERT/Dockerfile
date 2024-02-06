ARG FROM_IMAGE_NAME=jfrog.tecorigin.net/tecotp-docker/release/ubuntu22.04/x86_64/pytorch:0.12.0-torch_sdaa0.15.0
FROM ${FROM_IMAGE_NAME}

WORKDIR /workspace/
ADD requirements.txt /workspace/
RUN rm -rf /bin/sh && ln -s /bin/bash /bin/sh
RUN source activate torch_env && pip install --no-cache-dir -r requirements.txt
RUN sudo apt-get install libxml-parser-perl
ADD . /workspace/NLP/BERT/
WORKDIR /workspace/NLP/BERT/
