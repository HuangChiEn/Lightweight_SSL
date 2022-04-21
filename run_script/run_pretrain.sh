#!/bin/bash
# Description : this script is for conducting the pretrain-phase of SSL,
#               you can also passsing the argument to overwrite the config settings.
export CONFIGER_PATH="./ini/pretrain/simclr.ini"
python3 ../pretrain_proc.py #\
# plz feel free to to pass the commendline args.. if you need it ~ ~