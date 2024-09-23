FROM quay.io/jupyter/scipy-notebook
LABEL authors="msnp"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

USER ${NB_UID}

RUN mamba install --yes jupyter_kernel_gateway ipykernel matplotlib numpy nltk && \
    mamba clean --all -f -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

RUN python -m nltk.downloader punkt

COPY .env_docker .env

# Uncomment and set your token value here
# ENV TOKEN your_token_value_here
CMD ["jupyter", "notebook", \
    "--ip=0.0.0.0", \
    "--port=8888", \
    "--NotebookApp.token=''", \
    "--NotebookApp.password=''", \
    "--NotebookApp.allow_origin='*'", \
    "--NotebookApp.notebook_dir=${HOME}", \
    "--NotebookApp.open_browser=False", \
    "--NotebookApp.allow_remote_access=True", \
    "--FileContentsManager.allow_hidden=True"]

EXPOSE 8888

WORKDIR "${HOME}"
