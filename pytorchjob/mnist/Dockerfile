FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime



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

RUN pip install tensorboardX==1.6.0 protobuf==3.20.0

ADD mnist.py /workspace/mnist/src/mnist.py


ENTRYPOINT ["python","/workspace/mnist/src/mnist.py"]

