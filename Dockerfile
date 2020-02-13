FROM python:3.7
WORKDIR /src
COPY . .
RUN pip3 install .
RUN pip3 install pytest pylint
ENTRYPOINT ["python3"]
