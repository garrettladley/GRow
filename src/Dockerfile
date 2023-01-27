FROM ubuntu:latest

WORKDIR /code

COPY . .

RUN apt-get update 
RUN apt-get -yu dist-upgrade -y

RUN apt-get install -y golang-go 

RUN apt-get install -y python3.10 
RUN apt-get install -y python3-pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN python3.10 -m pip install --upgrade pip
RUN python3.10 -m pip install --no-cache-dir -r python/requirements.txt
RUN apt-get install -y tesseract-ocr

ENV CGO_ENABLED=0 GOOS=linux GOARCH=amd64 SSL_CERT_DIR=/etc/ssl/certs

WORKDIR /code/go
RUN go mod download

EXPOSE 4000

CMD ["go", "run", "main.go"]