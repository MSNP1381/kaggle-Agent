FROM quay.io/jupyter/scipy-notebook
LABEL authors="msnp"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

USER ${NB_UID}

COPY ./notebook_requirements.txt ./requirements.txt

RUN python -m pip install -r requirements.txt


RUN python -m nltk.downloader punkt

COPY .env.docker .env

USER ${NB_UID}
RUN mamba install --yes jupyter_kernel_gateway ipykernel &&     mamba clean --all -f -y &&     fix-permissions "${CONDA_DIR}" &&     fix-permissions "/home/${NB_USER}"

ENV TOKEN="UNSET"
CMD python -m jupyter kernelgateway --KernelGatewayApp.ip=0.0.0.0     --KernelGatewayApp.port=8888     --KernelGatewayApp.auth_token="${TOKEN}"     --JupyterApp.answer_yes=true     --JupyterWebsocketPersonality.list_kernels=true

EXPOSE 8888

EXPOSE 8888


WORKDIR /home/jovyan/working
