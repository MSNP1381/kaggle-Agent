FROM quay.io/jupyter/scipy-notebook
LABEL authors="msnp"


SHELL ["/bin/bash", "-o", "pipefail", "-c"]


USER ${NB_UID}

RUN mamba install --yes jupyter_kernel_gateway ipykernel matplotlib numpy nltk &&\
mamba clean --all -f -y &&\
fix-permissions "${CONDA_DIR}" &&\
fix-permissions "/home/${NB_USER}"

RUN python -m nltk.downloader punkt

COPY .env_docker .env
ENV $(cat .env | xargs)

CMD python -m jupyter kernelgateway \
--KernelGatewayApp.ip=0.0.0.0 \
    --KernelGatewayApp.port=8888 \
    --KernelGatewayApp.auth_token="${TOKEN}" \
    --JupyterApp.answer_yes=true \
    --JupyterWebsocketPersonality.list_kernels=true
    
    
    
EXPOSE 8888


WORKDIR "${HOME}"


