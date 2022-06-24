# Lightweight-SSL 🚀🚀

### 1. How to run the code ?
#### makefile instruction
`make run`
#### run bashfile 
`cd run_script ; ./run_script.sh`
###### The config file is pointed by the `CONFIGER_PATH` global variable in shell.
###### The pretrain/linea_eval phase will be decided by the path of config file.

### 2. How to config each method ?
#### For example. 
<pre>
## training loop
# Epoch to start learning from, used when resuming
epoch_start = 0@int
# Total number of epochs
epochs = 100@int

# the special argument will be place into section "[ssl_args]"
[ssl_args]
# optimizer
# 0.3 * 2400/256 = 2.81
lr = 2.81@float
weight_decay = 1e-6@float

# scheduler
warmup_epochs = 10@int
# max_epoch == epochs
</pre>

### TODO List ✔️
1. Run simsiam imagenet baseline
2. Run simclr imagenet baseline
3. Run barlow_twin baseline
4. Complete byol model

### Any suggestion & feedback 
📧 plz contact me : a3285556aa@gmail.com