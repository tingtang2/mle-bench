FROM mlebench-env

# where to put submission.csv, will be extracted
ARG SUBMISSION_DIR
ENV SUBMISSION_DIR=${SUBMISSION_DIR}
# where to put any logs, will be extracted
ARG LOGS_DIR
ENV LOGS_DIR=${LOGS_DIR}
# where to put any code, will be extracted
ARG CODE_DIR
ENV CODE_DIR=${CODE_DIR}
# where to put any other agent-specific files, will not be necessarily extracted
ARG AGENT_DIR
ENV AGENT_DIR=${AGENT_DIR}

RUN mkdir ${LOGS_DIR} ${CODE_DIR} ${AGENT_DIR}

ARG CONDA_ENV_NAME=agent
ARG REQUIREMENTS=${AGENT_DIR}/requirements.txt

# copy just the requirements file, so that we can cache conda separately from the agent files
COPY mle-bench/agents/rdagent_ds/requirements.txt ${AGENT_DIR}/requirements.txt

# Requirements for RD-Agent
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 git strace -y

# Copy RD-Agent codebase from mle-bench submodule (current version with EnsembleCoSTEER)
COPY RD-Agent ${AGENT_DIR}/RD-Agent

# Patch RD-Agent to skip download when data exists (for MLE-bench compatibility)
COPY mle-bench/agents/rdagent_ds/patch_rd_agent.py /tmp/patch_rd_agent.py
RUN python /tmp/patch_rd_agent.py

# Install RD-Agent from local directory (with version set to avoid setuptools-scm error)
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0
RUN conda run -n ${CONDA_ENV_NAME} bash -c "cd ${AGENT_DIR}/RD-Agent && pip install -e . && pip install --upgrade openai httpx anthropic coverage" && \
    conda clean -afy

# Create a 'kaggle' conda environment (RD-Agent expects this name for DS execution)
# Clone the 'agent' environment to create 'kaggle' env (will include coverage from agent env)
RUN conda create -n kaggle --clone ${CONDA_ENV_NAME} && \
    conda clean -afy

# put all the agent files in the expected location
COPY mle-bench/agents/rdagent_ds ${AGENT_DIR}/wrapper

# Give nonroot write access to agent dir (RD-Agent writes selector.log, etc.)
RUN mkdir -p ${AGENT_DIR}/rd_data ${AGENT_DIR}/rd_workspace && \
    chown -R nonroot:nonroot ${AGENT_DIR}

# MLEBench always runs `/home/agent/start.sh` inside the container.
# Our wrapper lives under `${AGENT_DIR}/wrapper`, so create a stable entrypoint path.
RUN ln -sf ${AGENT_DIR}/wrapper/start.sh ${AGENT_DIR}/start.sh
