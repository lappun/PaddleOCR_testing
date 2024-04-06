FROM nvcr.io/nvidia/tensorflow:21.10-tf1-py3 

RUN apt update; apt install -y build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev curl git vim \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev \
python3 python3-pip python3-venv

RUN useradd -ms /bin/bash paddle
RUN usermod -aG video paddle

ADD files /tmp/

RUN apt install -y python3-pip libopenblas-dev libjpeg-dev libpng-dev
RUN /tmp/setup-pytorch.sh
RUN dpkg -i /tmp/libcudnn8_8.4.1.50-1+cuda11.4_arm64.deb
RUN dpkg -i /tmp/libcudnn8-dev_8.4.1.50-1+cuda11.4_arm64.deb
