# Set the base image
FROM continuumio/miniconda3:latest AS builder

# Install system dependencies
RUN apt-get update && \
    apt-get install -y sudo libusb-1.0 gcc g++ python3-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /home/chimerabot

# Create conda environment
COPY setup/environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && \
    conda clean -afy && \
    rm /tmp/environment.yml

# Copy remaining files
COPY bin/ bin/
COPY chimerabot/ chimerabot/
COPY scripts/ scripts/
COPY controllers/ controllers/
COPY scripts/ scripts-copy/
COPY setup.py .
COPY LICENSE .
COPY README.md .

# activate chimerabot env when entering the CT
SHELL [ "/bin/bash", "-lc" ]
RUN echo "conda activate chimerabot" >> ~/.bashrc

COPY setup/pip_packages.txt /tmp/pip_packages.txt
RUN python3 -m pip install --no-deps -r /tmp/pip_packages.txt && \
    rm /tmp/pip_packages.txt


RUN python3 setup.py build_ext --inplace -j 8 && \
    rm -rf build/ && \
    find . -type f -name "*.cpp" -delete


# Build final image using artifacts from builder
FROM continuumio/miniconda3:latest AS release

# Dockerfile author / maintainer
LABEL maintainer="ChimeraBot Team <dev@chimerabot.org>"

# Build arguments
ARG BRANCH=""
ARG COMMIT=""
ARG BUILD_DATE=""
LABEL branch=${BRANCH}
LABEL commit=${COMMIT}
LABEL date=${BUILD_DATE}

# Set ENV variables
ENV COMMIT_SHA=${COMMIT}
ENV COMMIT_BRANCH=${BRANCH}
ENV BUILD_DATE=${BUILD_DATE}

ENV INSTALLATION_TYPE=docker

# Install system dependencies
RUN apt-get update && \
    apt-get install -y sudo libusb-1.0 && \
    rm -rf /var/lib/apt/lists/*

# Create mount points
RUN mkdir -p /home/chimerabot/conf /home/chimerabot/conf/connectors /home/chimerabot/conf/strategies /home/chimerabot/conf/controllers /home/chimerabot/conf/scripts /home/chimerabot/logs /home/chimerabot/data /home/chimerabot/certs /home/chimerabot/scripts /home/chimerabot/controllers

WORKDIR /home/chimerabot

# Copy all build artifacts from builder image
COPY --from=builder /opt/conda/ /opt/conda/
COPY --from=builder /home/ /home/

# Setting bash as default shell because we have .bashrc with customized PATH (setting SHELL affects RUN, CMD and ENTRYPOINT, but not manual commands e.g. `docker run image COMMAND`!)
SHELL [ "/bin/bash", "-lc" ]

# Set the default command to run when starting the container

CMD conda activate chimerabot && ./bin/chimerabot_quickstart.py 2>> ./logs/errors.log
