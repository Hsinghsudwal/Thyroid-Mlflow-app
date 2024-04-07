# app-Dockerfile

FROM python:3.8-alpine

COPY . /app

WORKDIR /app

RUN apt-get update && apt-get install -y

RUN pip install -r requirements.txt

EXPOSE 8000

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8000", "--server.address=0.0.0.0"]
