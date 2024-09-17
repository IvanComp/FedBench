FROM python:3.9-slim

WORKDIR /app

COPY ./client /app

RUN pip install -r requirements.txt

CMD ["python", "client.py"]
