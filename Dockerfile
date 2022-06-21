FROM python:3.9.12-slim

WORKDIR /var/server

COPY setup.py setup.py
COPY setup.cfg setup.cfg

COPY src src
COPY assets assets

RUN pip install . --no-cache-dir

CMD python src/remla/serve_model.py
