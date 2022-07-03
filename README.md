# Lightweight-SSL 🚀🚀
#### As the name implies, this project aims to provide an lightweight framework for building various kind of self-supervised learning (SSL) method, and benchmarking the current SOTA. Thanks for the famous repository **[solo-learn](https://github.com/vturrisi/solo-learn)** to provide the 'readable' prototype as the reference for us. 
> However, the repository contains too much feature and the small-scale researcher may not need it. So, I try my best to reduce & write an lightweight repository for the researcher. Especially, the foxconn SSL group. Hope you guys like it!!

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

### 3. Where to place the dataset ?
##### Thanks for the dockerization, all of dataset will be placed under the /data folder. For torch-dataset (online avaliable), all dataset will be download at /data/tch_ds. For the imagenet, i just place the dataset at the /data/ImgNet.

---

### TODO List ✔️
0. Build SOTA SSL methods
> - [ ] BYOL
> - [ ] MoCov2
> - [ ] Fix bugs (low acc of NNCLR, LARS not work in fine-tuning, scheduler..etc)

1. Baseline checking 🚀
> - [ ] Check Simsiam linear_eval performance on ImageNet (acc1. 59.4%, not good)
> - [ ] Check SimCLR linear_eval performance on ImageNet

2. Support various kind of backbone
> - [ ] Replace the backbone hub from torch_hub to timm
> - [ ] Try to built ViT

3. Support faster Trainning
> - [ ] Build FFCV dataset & dataloader
> - [ ] Build multi-node support (maybe based on pytorch-lightning)

### Any suggestion & feedback 
📧 plz contact me : a3285556aa@gmail.com