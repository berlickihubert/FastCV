FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt-get update && \
    apt-get install -y build-essential cmake g++ libx11-dev libjpeg-dev libpng-dev libtiff-dev libgif-dev file

COPY . /app
COPY resources /app/resources
WORKDIR /app


RUN cmake -Bbuild -H. && cmake --build build

CMD ["/app/build/main"]
