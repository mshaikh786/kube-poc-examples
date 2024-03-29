# Use the respective Makefile to pass the appropriate BASE_IMG and build the image correctly
ARG BASE_IMG=<base>
FROM $BASE_IMG

USER root

# args - software versions
ARG MINIFORGE_ARCH="x86_64"
 # renovate: datasource=github-tags depName=conda-forge/miniforge versioning=loose
ARG MINIFORGE_VERSION=23.3.1-0
ARG CONDA_VERSION=23.9.0
ARG PIP_VERSION=21.1.2
ARG PYTHON_VERSION=3.9.18

# install -- node.js
RUN export DEBIAN_FRONTEND=noninteractive \
 && curl -sL "https://deb.nodesource.com/gpgkey/nodesource.gpg.key" | apt-key add - \
 && echo "deb https://deb.nodesource.com/node_14.x focal main" > /etc/apt/sources.list.d/nodesource.list \
 && apt-get -yq update \
 && apt-get -yq install --no-install-recommends \
    nodejs \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# setup environment for conda
ENV CONDA_DIR /opt/conda
ENV CONDA_PKGS_DIRS=${HOME}/conda_cache
ENV PATH "${CONDA_DIR}/bin:${PATH}"
RUN mkdir -p ${CONDA_DIR} \
 && echo ". /opt/conda/etc/profile.d/conda.sh" >> ${HOME}/.bashrc \
 && echo ". /opt/conda/etc/profile.d/conda.sh" >> /etc/profile \
 && echo "conda activate base" >> ${HOME}/.bashrc \
 && echo "conda activate base" >> /etc/profile \
 && chown -R ${NB_USER}:${NB_GROUP} ${CONDA_DIR} \
 && chown -R ${NB_USER}:${NB_GROUP} ${HOME}

# switch to NB_UID for installs
USER ${NB_UID}

# install - conda, pip, python
RUN curl -sL "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-${MINIFORGE_ARCH}.sh" -o /tmp/Miniforge3.sh \
 && curl -sL "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-${MINIFORGE_ARCH}.sh.sha256" -o /tmp/Miniforge3.sh.sha256 \
 && echo "$(cat /tmp/Miniforge3.sh.sha256 | awk '{ print $1; }') /tmp/Miniforge3.sh" | sha256sum --check \
 && rm /tmp/Miniforge3.sh.sha256 \
 && /bin/bash /tmp/Miniforge3.sh -b -f -p ${CONDA_DIR} \
 && rm /tmp/Miniforge3.sh \
# RUN curl -sL "https://github.com/conda-forge/miniforge/releases/download/23.3.1-0/Mambaforge-23.3.1-0-Linux-x86_64.sh" -o /tmp/Mambaforge.sh \
# &&  /bin/bash /tmp/Mambaforge.sh  -b -f -p ${CONDA_DIR} \
#  && rm /tmp/Mambaforge.sh  \
 && mamba install -y -q \
    python=${PYTHON_VERSION} \
    pip=${PIP_VERSION} \
    conda=${MINIFORGE_VERSION} \
 && mamba update -y -q --all \
 && mamba clean -a -f -y \
 && chown -R ${NB_USER}:${NB_GROUP} ${CONDA_DIR} \
 && chown -R ${NB_USER}:${NB_GROUP} ${HOME} \
 && conda config --system --set auto_update_conda false \
 && conda config --system --set show_channel_urls true \
 && echo "conda ${CONDA_VERSION}" >> ${CONDA_DIR}/conda-meta/pinned \
 && echo "python ${PYTHON_VERSION}" >> ${CONDA_DIR}/conda-meta/pinned 

# install - requirements.txt
COPY --chown=${NB_USER}:${NB_GID} requirements.txt /tmp
RUN mamba install -y -c conda-forge -v jupyterlab=3.6 notebook=6.5 ipykernel=6.25 nb_conda_kernels nodejs\
 && python3 -m pip install -r /tmp/requirements.txt --quiet --no-cache-dir \
 && rm -f /tmp/requirements.txt \
 && jupyter lab --generate-config \
 && mamba clean -a -f -y \
 && rm -rf ${HOME}/.cache/yarn \
 && chown -R ${NB_USER}:${NB_GROUP} ${CONDA_DIR} \
 && chown -R ${NB_USER}:${NB_GROUP} ${HOME}

# s6 - copy scripts
COPY --chown=${NB_USER}:${NB_GROUP} s6/ /etc

# s6 - 01-copy-tmp-home
USER root
RUN mkdir -p /tmp_home \
 && cp -r ${HOME} /tmp_home \
 && chown -R ${NB_USER}:${NB_GROUP} /tmp_home
USER ${NB_UID}

EXPOSE 8888

ENTRYPOINT ["/init"]
