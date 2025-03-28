#!/bin/bash

cd /rhome/lhoellein/git/scannetpp/; python -m dslr.undistort dslr/configs/undistort.yml --stride $1 --offset $2