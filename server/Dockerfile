FROM python:3.9-slim

WORKDIR /app

COPY ./server /app

RUN pip install -r requirements.txt

CMD ["python", "server.py"]
