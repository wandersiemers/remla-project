FROM python:3.9

COPY requirements.txt /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt

COPY src /var/server/src
COPY assets /var/server/assets

WORKDIR /var/server

CMD python /var/server/src/serve_model.py