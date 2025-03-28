#!/bin/bash

cd /rhome/lhoellein/git/scannetpp/; python -m iphone.prepare_iphone_data iphone/configs/prepare_iphone_data.yml --stride $1 --offset $2
cd /rhome/lhoellein/git/scannetpp/; python -m iphone.undistort_iphone iphone/configs/undistort_iphone.yml --stride $1 --offset $2