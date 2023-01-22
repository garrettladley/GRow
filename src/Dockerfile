FROM ubuntu:latest

RUN apt-get update && apt-get -yu dist-upgrade -y \
    && apt-get install -y golang-go && apt-get install -y python3.10 \
    && apt-get install -y python3-pip && apt-get install -y tesseract-ocr \
    && python3.10 -m pip install --upgrade pip && python3.10 -m pip install --no-cache-dir -r python/requirements.txt

ENV GOROOT=/usr/lib/go
ENV GOPATH=$HOME/go
ENV PATH=$GOPATH/bin:$GOROOT/bin:$PATH

WORKDIR /code

COPY . /code

EXPOSE 4000

CMD ["go", "run", "go/main.go"]