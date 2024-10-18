FROM quay.io/jupyter/scipy-notebook
LABEL authors="msnp"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

USER ${NB_UID}

COPY ./notebook_requirements.txt ./requirements.txt

RUN python -m pip install -r requirements.txt


RUN python -m nltk.downloader punkt

COPY .env.docker .env

# Uncomment and set your token value here
# ENV TOKEN your_token_value_here
CMD ["jupyter", "notebook", \
    "--ip=0.0.0.0", \
    "--port=8888", \
    "--NotebookApp.token=''", \
    "--NotebookApp.password=''", \
    "--NotebookApp.allow_origin='*'", \
    "--NotebookApp.open_browser=False", \
    "--FileContentsManager.allow_hidden=True"]

EXPOSE 8888


WORKDIR /home/jovyan/working
