FROM python:3.7-bullseye
WORKDIR /usr/src/app
RUN pip install tensorflow==1.15.5

COPY ./TensorFlow/ ./

CMD ["/bin/bash", "./run-tensorflow.sh", "relu", "medium"]