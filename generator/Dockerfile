FROM python:3.9.12-slim

COPY requirements.txt /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt --no-cache-dir

COPY generate_events.py /var/app/generate_events.py

CMD python /var/app/generate_events.py
