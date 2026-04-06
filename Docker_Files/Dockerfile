FROM  python:3.12-slim-bookworm

RUN mkdir /app

COPY Ds-Project-1/ /app

WORKDIR /app   

RUN apt-get update -y && apt-get upgrade -y && apt-get install python3-pip -y


RUN pip3 install -r requirements.txt --no-cache-dir

RUN chmod +x start-backend-frontend.sh


CMD ["bash","start-backend-frontend.sh"]









