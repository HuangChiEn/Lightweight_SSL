# Lightweight-SSL 🚀🚀
#### As the name implies, this project aims to provide an lightweight framework for building various kind of self-supervised learning (SSL) method, and benchmarking the current SOTA. Thanks for the famous repository **[solo-learn](https://github.com/vturrisi/solo-learn)** to provide the "*readable*" prototype as the reference for us. 
> However, the repository contains too much feature and the small-scale experiments may not need it. So, I try my best to reduce & write an lightweight repository for the researcher. Especially, the foxconn SSL group. Hope you guys like it!!

## 1. How to run the code ?
#### Every repo should be easy to allow the user run the code, so I have prepared 2 options, one for the **make** lover, the other one for **bash** lover.. hope to enjoy it!!

### 🐮 makefile instruction
`make run`
### 🖋️ run bashfile 
`cd run_script ; ./run_script.sh`
> ##### The config file is pointed by the `CONFIGER_PATH` global variable in shell.
> ##### The pretrain/linea_eval phase will be decided by the path of config file.

---

## 2. How to config each method ?
#### Open the config file, all parameters are well-arranged into each section, however, what kind of parameters is necessary ?

### **Model architecture & Params feed into**
```
# Backbone: resnet18, 32, 50
backbone = resnet50@str
ssl_method = simclr@str

## the special argument will be place into section "[ssl_args]"
[ssl_args]
# optimizer
lr = 2.81@float
weight_decay = 1e-6@float
# scheduler
warmup_epochs = 10@int

# projector
proj_hidden_dim = 2048@int
proj_output_dim = 128@int

# loss : smaller value will force the embedding distribution become more compact!
temperature = 0.2@float
```

### **Training loop**
```
# Total number of epochs
epochs = 100@int

# global batch size
batch_size = 820@int

[ckpt_args]
dirpath = /workspace/meta_info/ckpts@str
# the filename follow the torchlight f-string rule
filename = simclr-{epoch}-{lin_acc1:.2f}@str
every_n_epochs = 25@int
save_top_k = -1@int
monitor = lin_acc1@str
```

### **Logger**
```
## logger (default CSV logger, we only need to define log_path and proj_name)
log_type = wandb@str

[wandb_args]
save_dir = /workspace/meta_info@str
name = simclr@str
project = lw_ssl@str
entity = josef@str
offline = @bool 
```

### **Dataset stuff**
```
## Dataset: cifar10|100, imagenet
dataset = imagenet@str
data_dir = /data/ImgNet/train@str

## The number of crop is the element of list, 
##     and the number of view is the len of list (default : 1 view 2 crop)
aug_crop_lst = [2]@list

# custmized dataset (default : False)
custom_ds_cfg = {}@dict

# Number of torchvision workers used to load data (default: 8)
num_workers = 18@int
```

### **Miscellaneous**
```
# Seed for Numpy and PyTorch. Default: -1 (None)
seed = 1@int

# Number of torchvision workers used to load data (default: 8)
num_workers = 18@int

# custmized dataset (default : False)
custom_ds_cfg = {}@dict

# logger (default CSV logger, we only need to define log_path and proj_name)
log_type = wandb@str
```

---

## 3. Where to place the dataset ?
##### Thanks for the dockerization, all of dataset will be placed under the /data folder. For torch-dataset (online avaliable), all dataset will be download at /data/tch_ds. For the imagenet, i just place the dataset at the /data/ImgNet.

---

### TODO List ✔️
0. Build SOTA SSL methods (🚀working on here..)
> - [ ] BYOL
> - [ ] MoCov2
> - [ ] Fix bugs (low acc of NNCLR, LARS not work in fine-tuning, scheduler..etc)

1. Baseline checking 
> - [x] Check Simsiam linear_eval performance on ImageNet
> - [x] Check SimCLR linear_eval performance on ImageNet
> - [ ] Check BarlowTwin linear_eval performance on ImageNet

2. Support various kind of backbone
> - [ ] Replace the backbone hub from [torch_hub](https://pytorch.org/hub/) to [timm](https://github.com/rwightman/pytorch-image-models)
> - [ ] Try to built ViT

3. Support faster Trainning
> - [ ] Build FFCV dataset & dataloader
> - [ ] Build multi-node support (maybe based on pytorch-lightning)

---

### Reproduct benchmark performance :
#### The expected performance of linear evaluation under 100 epoch pretrained is described as the [simsiam paper](https://arxiv.org/abs/2011.10566), except the [SimCLR](https://arxiv.org/pdf/2002.05709.pdf) performance is almost match with the Appendix B.1. Actually the other paper only reveal 800 epoch pretrained performance, which is not suitable for correcteness verification though.

#### The evaluation protocol : 
> Since most of the method have slightly different in their fine-tuning protocol, we attempt to **unified** the protocol we use. The following table present the performance under 100 epoch pretraining with their corresponding setup. However, the fine-tuning method, we always apply LAMB optimizer with the cosin-anneling scheduler with 90 epoch training. Furthermore, we load the fine-tuned model & call the test method of trainer as the suggestion of pytorch-lighting, instead of call the fit method which re-use the train_step..

|             | pretrain top-1 lin_acc | eval top-1 lin_acc | eval top-5 lin_acc | pretra_bkbone ckpt | lin_clf ckpt |
|:-----------:|:----------------------:|:------------------:|:------------------:|:------------------:|:------------:|
| **Simsiam** |            26.07%           |          76.99%         |          93.10%         |          [🔗](https://drive.google.com/file/d/1XRPgMK-zvYYTNSrY7i4lRD8CEZSUMcCw/view?usp=sharing)         |       [🔗](https://drive.google.com/file/d/14Va9Jd3Pq0nznldPnKBkfbTkaifsZDch/view?usp=sharing)      |
|  **SimCLR** |            38.70%           |          64.99%         |          85.94%         |          [🔗](https://drive.google.com/file/d/1OV7splqRlHoMuJCC8z-p_j_JyBDxsB6e/view?usp=sharing)         |       [🔗](https://drive.google.com/file/d/1XLdvQRU_TjtVIIalZlvedrdAG2tWhL7p/view?usp=sharing)      |
|  **Barlowtwin** |            24.92%           |          54.35%         |          76.72%         |          [🔗](https://drive.google.com/file/d/1DmlX2wnpCgEC9ufgigUxTcPRgyMcWmiz/view?usp=sharing)         |       [🔗](https://drive.google.com/file/d/1O-g_D9RN_u-okxyfwWSNR1GXpQewUnDH/view?usp=sharing)      |
|    NNCLR    |            ❌           |          ❌         |          ❌         |          ❌         |       ❌      |
|     BYOL    |            ❌           |          ❌         |          ❌         |          ❌         |       ❌      |
|    MoCov2   |            ❌           |          ❌         |          ❌         |          ❌         |       ❌      |

---

### Any suggestion & feedback 
📧 plz contact me : a3285556aa@gmail.com