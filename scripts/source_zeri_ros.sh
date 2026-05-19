#!/usr/bin/env bash
source /opt/ros/jazzy/setup.bash
source ~/NBYtics/.venv/bin/activate
source ~/NBYtics/install/setup.bash
export PYTHONPATH=$HOME/NBYtics/.venv/lib/python3.12/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
