FROM python:3.9.12-slim

WORKDIR /root/

COPY requirements.txt .

RUN mkdir output
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

COPY src src
COPY data data

# TODO Run the data processing scripts
# TODO run the inference server once ready
# EXPOSE 8080
# ENTRYPOINT ["python"]
# CMD ["src/serve_model.py"]
