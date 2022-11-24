#!/bin/bash
# Expose the X server on the host.
sudo xhost +local:root
# --rm: Make the container ephemeral (delete on exit).
# -it: Interactive TTY.
# --gpus all: Expose all GPUs to the container.
# access jupyter notebook via http://localhost:1234/
if [[ $(uname -m) == 'arm64' ]]; then
  docker run \
  --platform linux/amd64\
  --rm \
  -it \
  -v $(pwd):/sat \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -p 1234:8888 -p 6006:6006 \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  --privileged \
  --name sat_container sat-dev
else
  docker run \
  --rm \
  -it \
  --gpus all \
  -v $(pwd):/sat \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -p 1234:8888 -p 6006:6006 \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  --privileged \
  --name sat_container sat-dev
fi

