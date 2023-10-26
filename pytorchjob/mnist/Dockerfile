FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
RUN apt-get update && apt-get install -y curl unzip


ENV JOB_USER training01
ENV JOB_UID 800001
ENV JOB_GID 800001
ENV JOB_GROUP ${JOB_USER}
ENV HOME /home/$JOB_USER
ENV SHELL /bin/bash

RUN groupadd --gid ${JOB_GID} ${JOB_GROUP}
RUN groupadd --gid 1337 kfgroup
RUN useradd --no-log-init -M -s /bin/bash -u ${JOB_UID} -g ${JOB_GID} -G ${JOB_GROUP},kfgroup ${JOB_USER} \
&& mkdir -p ${HOME} \
&& chown -R ${JOB_USER}:${JOB_GROUP} ${HOME} \
&& chown -R ${JOB_USER}:${JOB_GROUP} /usr/local/bin 

RUN  mkdir -p /workspace/mnist/src \
  && chgrp -R kfgroup /workspace \
  && chmod -R g+rwx /workspace 


USER ${JOB_USER}

WORKDIR /data
RUN curl -sL "http://cs231n.stanford.edu/tiny-imagenet-200.zip" -o /tmp/tiny-imagenet-200.zip \ 
&& unzip /tmp/tiny-imagenet-200.zip \
&& rm -r /tmp/tiny-imagenet-200.zip
ENV DATA_DIR=/data/tiny-imagenet-200

RUN pip install tensorboardX==1.6.0 protobuf==3.20.0 matplotlib

ADD mnist_v2.py /workspace/mnist/src/mnist_v2.py
ADD mnist.py /workspace/mnist/src/mnist.py
ADD ddp.py /workspace/mnist/src/ddp.py
ADD device_info.py /workspace/mnist/src/device_info.py

WORKDIR /workspace



