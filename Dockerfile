FROM ros:humble

# install dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-colcon-common-extensions \
    python3-vcstool \
    python3-rosdep \
    python3-pip

# install dependencies for the package
RUN pip3 install \
    onnxruntime \
    numpy \
    opencv-python

# create a directory for the workspace
RUN mkdir -p /ros2_ws/src

# copy the package to the workspace
COPY . /ros2_ws/src

# set the working directory
WORKDIR /ros2_ws

# build the workspace
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
    colcon build --symlink-install --parallel-workers 4"

# start the node
CMD ["/bin/bash", "-c", "source /ros2_ws/install/setup.bash && ros2 run mapper_speedrun mapper"]
