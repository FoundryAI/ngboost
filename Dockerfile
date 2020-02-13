FROM python:3.7
WORKDIR /src
COPY . .
RUN pip3 install -r requirements.txt
ENTRYPOINT ["python3"]
