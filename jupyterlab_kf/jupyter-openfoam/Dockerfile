# Use the respective Makefile to pass the appropriate BASE_IMG and build the image correctly
ARG BASE_IMG=<base>
FROM $BASE_IMG

USER root

# args - software versions
ARG MINIFORGE_ARCH="x86_64"
 # renovate: datasource=github-tags depName=conda-forge/miniforge versioning=loose
ARG MINIFORGE_VERSION=4.10.1-4
ARG PIP_VERSION=21.1.2
#ARG PYTHON_VERSION=3.8.10
ARG PYTHON_VERSION=3.9.10
ARG OF_VERSION=2012

# Jupyter Lab installation

# install -- node.js
RUN export DEBIAN_FRONTEND=noninteractive \
 && curl -sL "https://deb.nodesource.com/gpgkey/nodesource.gpg.key" | apt-key add - \
 && echo "deb https://deb.nodesource.com/node_14.x focal main" > /etc/apt/sources.list.d/nodesource.list \
 && add-apt-repository ppa:rock-core/qt4 \
 && apt-get install -y \
 && apt-get -yq install --no-install-recommends \
    nodejs build-essential libfl-dev libbison-dev cmake zlib1g-dev \
    gnuplot libreadline-dev libncurses-dev libxt-dev \
    gcc-10 g++-10 gfortran-10 curl make libnuma-dev libnl-3-200 autoconf \
    qt4-dev-tools libqt4-dev libqt4-opengl-dev freeglut3-dev libqtwebkit-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 7 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 7 && \
    update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-10 7


# setup environment for conda
ENV CONDA_DIR /opt/conda
ENV PATH "${CONDA_DIR}/bin:/usr/local/bin:${PATH}"
ENV LD_LIBRARY_PATH "/usr/local/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
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
 && conda config --system --set auto_update_conda false \
 && conda config --system --set show_channel_urls true \
 && echo "conda ${MINIFORGE_VERSION:0:-2}" >> ${CONDA_DIR}/conda-meta/pinned \
 && echo "python ${PYTHON_VERSION}" >> ${CONDA_DIR}/conda-meta/pinned \
 && conda install -y -q \
    python=${PYTHON_VERSION} \
    conda=${MINIFORGE_VERSION:0:-2} \
    pip=${PIP_VERSION} \
 && conda update -y -q --all \
 && conda clean -a -f -y \
 && chown -R ${NB_USER}:${NB_GROUP} ${CONDA_DIR} \
 && chown -R ${NB_USER}:${NB_GROUP} ${HOME}

# install - requirements.txt
COPY --chown=${NB_USER}:${NB_GID} requirements.txt /tmp
RUN conda install -y -c conda-forge mamba \ 
 && mamba install -y -c conda-forge jupyterlab notebook ipykernel nb_conda_kernels nodejs\
 && python3 -m pip install -r /tmp/requirements.txt --quiet --no-cache-dir \
 && rm -f /tmp/requirements.txt \
 && jupyter lab --generate-config \
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

# This shall move up on top with all other system packages.
RUN apt-get update && apt-get install -y openssh-server


USER ${NB_UID}




# OpenFOAM-Installation

ENV OF_HOME="/apps/CFD/OpenFOAM-v${OF_VERSION}"
ENV OF_BASHRC="${OF_HOME}/etc/bashrc"


EXPOSE 8888

ENTRYPOINT ["/init"]
