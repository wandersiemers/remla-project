FROM python:3.9

ADD requirements.txt /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt

ADD src /var/server/src
ADD assets /var/server/assets

CMD python /var/server/src/serve_model.py
