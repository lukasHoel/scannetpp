#!/bin/bash

cd /rhome/lhoellein/git/scannetpp/; python -m iphone.prepare_iphone_data iphone/configs/prepare_iphone_data.yml --stride $1 --offset $2