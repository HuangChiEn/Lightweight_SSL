#!/bin/bash
# Description : this script is for conducting the pretrain-phase of SSL,
#               you can also passsing the argument to overwrite the config settings.
export CONFIGER_PATH="./linear_eval/simsiam.ini"
python3 ../linear_eval.py #\
# plz feel free to to pass the commendline args.. if you need it ~ ~