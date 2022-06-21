# # # this is our first build stage, it will not persist in the final image
# FROM ubuntu:latest

# # install git
# RUN apt update -y
# RUN apt upgrade -y
# # RUN apt-get install python3-pip -y

# RUN apt install software-properties-common -y
# RUN apt update

# # RUN add-apt-repository ppa:deadsnakes/ppa
# RUN apt update
# RUN apt install python3.8


# # install git
# RUN sudo apt install git -y

# RUN python3.8 -m pip install pip --upgrade

# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.9-slim

# RUN apk update
# RUN apk add git

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME

# Install regular packages
COPY requirements.txt .
RUN pip install -r requirements.txt

RUN pip instal gunicorn
# First create submodule using the below command
# git submodule add https://github.com/kalpesh22-21/haystack _submodules/haystack
# Install submodule packages
COPY _submodules/haystack _submodules/haystack
RUN pip install _submodules/haystack

COPY . ./
# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling.
CMD exec gunicorn --bind :$PORT --workers 4 --threads 8 --timeout 0 main:app