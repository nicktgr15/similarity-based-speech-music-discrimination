FROM ubuntu:14.04

# Install Yaafe
RUN apt-get update
RUN apt-get install cmake cmake-curses-gui libargtable2-0 libargtable2-dev libsndfile1 libsndfile1-dev libmpg123-0 libmpg123-dev libfftw3-3 libfftw3-dev liblapack-dev libhdf5-serial-dev libhdf5-7 wget build-essential git python -y

RUN git clone https://github.com/Yaafe/Yaafe.git yaafe
RUN cd yaafe && git checkout tags/v0.65
RUN mkdir yaafe/build
RUN cd yaafe/build && cmake ..
RUN cd yaafe/build && make
RUN cd yaafe/build && make install

ENV DEST_DIR /usr/local
ENV YAAFE_PATH $DEST_DIR/yaafe_extensions
ENV PATH $PATH:$DEST_DIR/bin
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:$DEST_DIR/lib
ENV PYTHONPATH $PYTHONPATH:$DEST_DIR/python_packages

# Install ffmpeg
RUN apt-get install software-properties-common python-software-properties -y
RUN add-apt-repository ppa:mc3man/trusty-media -y
RUN apt-get update
RUN apt-get dist-upgrade -y
RUN apt-get install ffmpeg -y

# Install youtube-dl
RUN apt-get install curl -y
RUN curl https://yt-dl.org/downloads/2015.08.28/youtube-dl -o /usr/local/bin/youtube-dl
RUN chmod a+rx /usr/local/bin/youtube-dl

# APP
RUN apt-get install python-pip python-dev -y
RUN apt-get install libfreetype6-dev -y
RUN apt-get install gfortran -y
RUN apt-get install python-tk -y

RUN pip install --upgrade pip
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY datasets/featureplans/featureplan /opt/speech-music-discrimination/featureplan
COPY src /opt/speech-music-discrimination
