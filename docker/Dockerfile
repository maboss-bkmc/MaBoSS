FROM ubuntu:bionic

MAINTAINER Vincent Noel version: 1.0

# Preventing python3-tk from asking timezone
ENV DEBIAN_FRONTEND=noninteractive

# Installing dependencies
RUN apt-get -qq update
RUN apt-get install -yq git flex bison gcc g++ python python-pip python3 python3-pip python3-tk
RUN pip3 install matplotlib pandas seaborn xlsxwriter

# Installing MaBoSS
ADD . /opt/MaBoSS-env-2.0
RUN bash -c "cd /opt/MaBoSS-env-2.0; ./check-requirements.sh"
RUN bash -c "cd /opt/MaBoSS-env-2.0/engine/src/; make install"

# Exposing port 7777
EXPOSE 7777

# Running MaBoSS server
CMD /opt/MaBoSS-env-2.0/engine/pub/MaBoSS-server --host 0.0.0.0 --port 7777
