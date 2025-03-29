#!/bin/bash

cd /rhome/lhoellein/git/scannetpp/; python -m common.render common/configs/render.yml --stride $1 --offset $2 --max_iter 1