#!/bin/bash

cd /rhome/lhoellein/git/scannetpp/; python -m iphone.undistort_iphone iphone/configs/undistort_iphone.yml --stride $1 --offset $2