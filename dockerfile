FROM python:3.11.0

WORKDIR /tmp

RUN pip install --upgrade pip

COPY requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir --upgrade -r requirements.txt

#COPY .env /tmp/.env

COPY app.py /tmp/app.py

WORKDIR /tmp

CMD ["python","app.py"]
