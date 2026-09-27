# Use the NVIDIA CUDA base environment
FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV TZ=UTC
ENV RUNS_IN_DOCKER=true
USER root
# Install GUI SUPPORT and Common Dependencies
RUN apt-get update && apt-get install -y \
    locales wget curl gnupg2 git git-lfs bash-completion build-essential \
    cmake vim python3-pip python3-virtualenv tree sudo tmux tmuxp unzip \
    lsb-release nano openssh-client python3-argcomplete software-properties-common \
    gfortran libx11-dev \
    && locale-gen en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
# Install ROS 2 Humble , GUI support, gazebo harmonic
RUN apt-get update && apt-get install -y \
    libgl1 libgl1-mesa-dri libglx-mesa0 x11-apps x11-utils xauth \
    && add-apt-repository universe \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
    && curl -sSL https://packages.osrfoundation.org/gazebo.gpg -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null \
    && apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-desktop \
    ros-humble-ament-* \
    ros-humble-ros-gzharmonic \
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-dev-tools \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --no-cache-dir --default-timeout=1000 \
    numpy opencv-python torch casadi empy==3.3.4 pyros-genmsg "setuptools<80" \
    pandas pytorch-mppi gymnasium

# Acados installation
WORKDIR /opt/acados
RUN git clone https://github.com/acados/acados.git . \
    && git submodule update --recursive --init \
    && mkdir -p build && cd build \
    && cmake -DACADOS_WITH_QPOASES=ON .. \
    && make install -j4
RUN pip3 install --no-cache-dir --upgrade pip wheel
RUN pip3 install --no-cache-dir ./interfaces/acados_template
ENV ACADOS_SOURCE_DIR=/opt/acados 
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/opt/acados/lib

# XRCE-DDS Agent installation
WORKDIR /opt
RUN git clone -b v2.4.3 https://github.com/eProsima/Micro-XRCE-DDS-Agent.git \
    && cd Micro-XRCE-DDS-Agent \
    && mkdir build && cd build \
    && cmake .. && make -j4 && make install \
    && ldconfig /usr/local/lib

# Install Tera Renderer for Acados C-code generation
RUN mkdir -p /opt/acados/bin && \
    wget https://github.com/acados/tera_renderer/releases/download/v0.2.1/t_renderer-v0.2.1-linux-amd64 -O /opt/acados/bin/t_renderer && \
    chmod +x /opt/acados/bin/t_renderer

# Install QGroundControl & optional dependencies
RUN apt-get update && apt-get install -y \
    gstreamer1.0-plugins-bad gstreamer1.0-libav gstreamer1.0-gl python3-gi python3-gst-1.0 \
    libfuse2 libxcb-xinerama0 libxkbcommon-x11-0 libxcb-cursor-dev \
    iputils-ping iproute2 acpi fontconfig pre-commit ros-humble-plotjuggler-ros \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY PX4-Autopilot/Tools/setup/requirements.txt /tmp/requirements.txt
COPY PX4-Autopilot/Tools/setup/ubuntu.sh /tmp/px4_setup.sh

# Run setup script (bypassing heavy ARM compilers and default sims)
RUN bash /tmp/px4_setup.sh
# --no-nuttx --no-sim-tools
# Create non-root user with passwordless sudo and dialout group access
ARG USER_NAME=rosuser
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid ${USER_GID} ${USER_NAME} \
&& useradd --shell /bin/bash --uid ${USER_UID} --gid ${USER_GID} -m ${USER_NAME} \
&& echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
&& usermod -aG dialout,video ${USER_NAME}

# Download QGroundControl AppImage and make it executable
RUN sudo wget -O /usr/local/bin/QGroundControl.AppImage https://github.com/mavlink/qgroundcontrol/releases/download/v5.0.8/QGroundControl-x86_64.AppImage \
# RUN sudo wget -O /usr/local/bin/QGroundControl.AppImage https://d176tv9ibo4jno.cloudfront.net/latest/QGroundControl-x86_64.AppImage \
    && sudo chmod +x /usr/local/bin/QGroundControl.AppImage \
    && cd /usr/local/bin \
    && ./QGroundControl.AppImage --appimage-extract \
    && rm /usr/local/bin/QGroundControl.AppImage \
    && sudo chmod -R a+rx /usr/local/bin/squashfs-root \
    && ln -s /usr/local/bin/squashfs-root/AppRun /usr/local/bin/qgroundcontrol

    USER ${USER_NAME}
    
    # uv installer (Pascal)
    # RUN curl -LsSf https://astral.sh/uv/install.sh | sh
    # install px4 requirements in the user side too so we can run make
RUN pip3 install --user --no-cache-dir -r /tmp/requirements.txt
ENV PATH="/home/${USER_NAME}/.local/bin:${PATH}"
ENV FASTDDS_BUILTIN_TRANSPORTS=UDPv4
RUN git config --global --add safe.directory '*'
RUN pip3 install --no-cache-dir "numpy>=1.17.3,<1.25.0"    

WORKDIR /ros2_ws
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
echo "source /ros2_ws/install/setup.bash" >> ~/.bashrc && \
echo 'alias build_mpc="cd /ros2_ws && colcon build --packages-up-to px4_mpc"' >> ~/.bashrc

CMD ["/bin/bash"]