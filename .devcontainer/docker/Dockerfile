ARG UBUNTU_VERSION=22.04

FROM ubuntu:${UBUNTU_VERSION} AS base

ARG WORKSPACE_DIR
ARG USERNAME
ARG UID
ARG GID

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV ROS_DISTRO=humble
ENV PATH="/home/${USERNAME}/.local/bin:$PATH"

WORKDIR ${WORKSPACE_DIR}

RUN apt-get update && apt-get install -y --no-install-recommends \
    locales \
    curl \
    gnupg2 \
    lsb-release \
    software-properties-common \
    ca-certificates \
    sudo \
    bash-completion \
    build-essential \
    cmake \
    git \
    gdb \
    python3-numpy \
    python3-dev \
    python3-pip \
    python3-tk \
    clang \
    clang-format \
    clang-tidy \
    && locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && add-apt-repository universe \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
    && ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}') \
    && curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb" \
    && dpkg -i /tmp/ros2-apt-source.deb \
    && rm /tmp/ros2-apt-source.deb \
    && apt-get update \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    ros-humble-desktop \
    ros-humble-rqt \
    ros-humble-rqt-common-plugins \
    ros-humble-rviz2 \
    ros-humble-tf2-tools \
    ros-humble-imu-filter-madgwick \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    python3-transforms3d \
    ros-humble-moveit \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-xacro \
    ros-humble-joint-state-publisher \
    ros-humble-joint-state-publisher-gui \
    ros-humble-robot-state-publisher \
    ros-humble-tf2-ros \
    ros-humble-tf2-geometry-msgs \
    ros-humble-ament-cmake-gtest \
    ros-humble-launch-testing \
    ros-humble-launch-testing-ament-cmake \
    ros-humble-moveit-ros-visualization \
    ros-humble-moveit-ros-move-group \
    ros-humble-moveit-planners-ompl \
    ros-humble-moveit-kinematics \
    ros-humble-moveit-simple-controller-manager \
    ros-humble-moveit-resources-panda-moveit-config \
    ros-humble-moveit-resources-panda-description \
    \
    # ros-humble-gazebo-ros-pkgs \
    # ros-humble-gazebo-plugins \
    # ros-humble-gazebo-ros2-control \
    \
    mesa-utils \
    x11-apps \
    wget \
    && rm -rf /var/lib/apt/lists/*


RUN python3 -m pip install --no-cache-dir \
    pettingzoo \
    pygame \
    robot_descriptions \
    viser \
    yourdfpy 
# RUN pip3 uninstall -y torch torchvision torchaudio
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cpu

RUN rosdep init \
    && rosdep fix-permissions \
    && rosdep update

RUN existing_user=$(getent passwd ${UID} | cut -d: -f1 || true) \
    && if [ -n "$existing_user" ] && [ "$existing_user" != "${USERNAME}" ]; then userdel -rf "$existing_user" || true; fi \
    && existing_group=$(getent group ${GID} | cut -d: -f1 || true) \
    && if [ -n "$existing_group" ] && [ "$existing_group" != "${USERNAME}" ]; then groupdel "$existing_group" || true; fi \
    && addgroup --gid ${GID} ${USERNAME} \
    && adduser --disabled-password --gecos '' --uid ${UID} --gid ${GID} ${USERNAME} \
    && echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME} \
    && mkdir -p /tmp/runtime-${USERNAME} \
    && chmod 700 /tmp/runtime-${USERNAME} \
    && chown -R ${UID}:${GID} /home/${USERNAME} /tmp/runtime-${USERNAME} ${WORKSPACE_DIR}

RUN echo "source /opt/ros/humble/setup.bash" >> /etc/bash.bashrc \
    && echo "export XDG_RUNTIME_DIR=/tmp/runtime-${USERNAME}" >> /etc/bash.bashrc

ENV DEBIAN_FRONTEND=dialog
USER ${USERNAME}
WORKDIR ${WORKSPACE_DIR}
CMD ["/bin/bash"]
