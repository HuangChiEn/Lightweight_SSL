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
> - [x] Check Simsiam linear_eval performance on ImageNet
> - [ ] Check SimCLR linear_eval performance on ImageNet

2. Support various kind of backbone
> - [ ] Replace the backbone hub from torch_hub to timm
> - [ ] Try to built ViT

3. Support faster Trainning
> - [ ] Build FFCV dataset & dataloader
> - [ ] Build multi-node support (maybe based on pytorch-lightning)

---

### Reproduct benchmark performance :
##### The expected performance of linear evaluation under 100 epoch pretrained is described as the [simsiam paper](https://arxiv.org/abs/2011.10566), except the [SimCLR](https://arxiv.org/pdf/2002.05709.pdf) performance is almost match with the Appendix B.1. Actually the other paper only reveal 800 epoch pretrained performance, which is not suitable for correcteness verification though.

##### The evaluation protocol : 
> Since most of the method have slightly different in their fine-tuning protocol, we attempt to **unified** the protocol we use. The following table present the performance under 100 epoch pretraining with their corresponding setup. However, the fine-tuning method, we always apply LAMB optimizer with the cosin-anneling scheduler with 90 epoch training. Furthermore, we load the fine-tuned model & call the test method of trainer as the suggestion of pytorch-lighting, instead of call the fit method which re-use the train_step..

|             | pretrain top-1 lin_acc | eval top-1 lin_acc | eval top-5 lin_acc | pretra_bkbone ckpt | lin_clf ckpt |
|:-----------:|:----------------------:|:------------------:|:------------------:|:------------------:|:------------:|
| **Simsiam** |            26.07%           |          76.99%         |          93.10%         |          [🔗](https://drive.google.com/file/d/1XRPgMK-zvYYTNSrY7i4lRD8CEZSUMcCw/view?usp=sharing)         |       [🔗](https://drive.google.com/file/d/14Va9Jd3Pq0nznldPnKBkfbTkaifsZDch/view?usp=sharing)      |
|    SimCLR   |            38.70%           |          64.99%         |          85.94%         |          [🔗](tmp)         |       [🔗](tmp)      |
|  Barlowtwin |            ❌           |          ❌         |          ❌         |          ❌         |       ❌      |
|    NNCLR    |            ❌           |          ❌         |          ❌         |          ❌         |       ❌      |
|     BYOL    |            ❌           |          ❌         |          ❌         |          ❌         |       ❌      |
|    MoCov2   |            ❌           |          ❌         |          ❌         |          ❌         |       ❌      |

### Any suggestion & feedback 
📧 plz contact me : a3285556aa@gmail.com