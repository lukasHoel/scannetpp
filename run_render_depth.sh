#!/bin/bash

s=$1
o=$2
for ((i = 0 ; i < 50 ; i++ ));
do
    cd /rhome/lhoellein/git/scannetpp/; python -m common.render common/configs/render.yml --stride $s --offset $o --max_iter 1
    o=$((o + s))
    echo $o
done