FROM ros:humble

# install dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-colcon-common-extensions \
    python3-vcstool \
    python3-rosdep \
    python3-pip \
    ros-humble-tf2-ros \
    ros-humble-sensor-msgs \
    ros-humble-geometry-msgs \
    ros-humble-foxglove-bridge \
    git \
    libgl1 \
    gdb

# install dependencies for the package
RUN pip3 install \
    onnxruntime-gpu \
    numpy \
    opencv-python \
    scipy

# create a directory for the workspace
RUN mkdir -p /ros2_ws/src

# install lart_msgs
WORKDIR  /ros2_ws/src/lart_msgs
RUN echo "hello"
RUN git clone -b dev https://github.com/FSLART/lart_msgs.git

# build lart_msgs
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
    colcon build --symlink-install --parallel-workers 4 --packages-select lart_msgs"

# copy the package to the workspace
COPY . /ros2_ws/src/mapper_speedrun

# build the NMS C/C++ implementation
RUN mkdir -p /ros2_ws/src/mapper_speedrun/mapper_speedrun/nms_core/build
WORKDIR /ros2_ws/src/mapper_speedrun/mapper_speedrun/nms_core/build
RUN g++ -shared -fPIC -o libnms.so ../nms.cpp

# set the working directory
WORKDIR /ros2_ws

# build the workspace
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
    colcon build --symlink-install --parallel-workers 4"

# start the node
CMD ["/bin/bash", "-c", "source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && ros2 launch /ros2_ws/src/mapper_speedrun/launch/mapper_launch.py"]
