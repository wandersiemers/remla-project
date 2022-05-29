FROM python:3-alpine

COPY requirements.txt /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt

COPY generate_events.py /var/app/generator.py

CMD python /var/app/generator.py
