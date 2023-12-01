FROM tensorflow/tensorflow:2.14.0-gpu
ENV TF_CPP_MIN_LOG_LEVEL=1
RUN apt-get update && \
  apt-get install -q -y \
  git \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/walline/ssl-tf2-sefoss
WORKDIR /ssl-tf2-sefoss
RUN pip install -r requirements.txt
