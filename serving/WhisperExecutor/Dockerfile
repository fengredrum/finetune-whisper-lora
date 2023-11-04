ARG JINA_VERSION=3.20.3-py310-standard

FROM jinaai/jina:${JINA_VERSION}

# install requirements before copying the workspace
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir --compile -r /requirements.txt

# setup the workspace
COPY . /workdir/
WORKDIR /workdir

# ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]