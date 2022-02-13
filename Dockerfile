from ubuntu:20.04


RUN apt-get -y update --fix-missing

RUN apt-get -y install make gcc g++ git wget vim curl openssh-server python3-pip

RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -

RUN apt-get -y update

RUN apt-get install ros-noetic-desktop-full
