FROM ubuntu:20.04
RUN apt-get update && apt-get install -y python3.8-dev python3.8-distutils wget
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 2

WORKDIR /tmp
RUN wget  https://bootstrap.pypa.io/get-pip.py && \
python get-pip.py
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
WORKDIR /
RUN rm /tmp/*


