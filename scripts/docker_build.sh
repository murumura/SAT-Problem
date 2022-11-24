#!/bin/bash
if [[ $(uname -m) == 'arm64' ]]; then
    docker build --platform linux/amd64 -t sat-dev .
else
    docker build -t sat-dev .
fi
