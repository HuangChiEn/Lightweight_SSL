from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DDPPlugin

from model import load_linear_clf, get_backbone
from datamanager import get_pretrain_datamanager
from util_tool.utils import get_wandb_logger


def pretrain_proc(cfger, DEFAULT_DATA_DIR='/data/tch_ds'):
    # 1. prepare dataset
    data_dir = cfger.data_dir if getattr(cfger, 'data_dir', None) else DEFAULT_DATA_DIR
    ds = get_pretrain_datamanager(cfger.dataset, data_dir, cfger.aug_crop_lst)
    if cfger.val_split:
        ds.setup(stage='train', cfger.val_split)  

    tra_ld = ds.train_dataloader(batch_size=cfger.batch_size, num_workers=cfger.num_workers)
    num_cls = ds.dataset_info.num_classes

    # 2. prepare SSL model
    res_net = get_backbone(cfger.backbone)
    ssl_model = wrap_ssl_method(res_net, num_cls, cfger.ssl_method, cfger.ssl_args)

    # 3. prepare logger
    logger = get_wandb_logger(**cfger.wandb_args)
    logger.log_hyperparams(vars(cfger))
    
    # 4.prepare callback : lr_monitor,  # model_ckpt
    callbk_lst = []
    get_model_ckpt(callbk_lst, **cfger.ckpt_args)
    get_lr_monitor(callbk_lst)

    # 4. prepare trainer config from prev def & conduct training loop
    tra_cfg = {'deterministic':bool(cfger.seed != -1), 'logger':logger, 'callbacks':callbk_lst, 
                'max_epochs':cfger.epochs, **cfger.compute_cfg}
    # setup find_unused_parameters=False to disable warning & speed-up
    tra_cfg['strategy'] = DDPPlugin(find_unused_parameters=False) if tra_cfg['strategy'] == 'ddp' else tra_cfg['strategy']

    trainer = Trainer(sync_batchnorm=True, **tra_cfg)
    trainer.fit(ssl_model, tra_ld)    
    if 'ckpt_path' in cfger:
        trainer.save_checkpoint(cfger.ckpt_path)


def linear_eval(cfger):
    # 1. prepare dataset
    data_dir = cfger.data_dir if getattr(cfger, 'data_dir', None) else '/data'
    ds = get_datamanager(cfger.dataset, data_dir, cfger.aug_crop_lst)
    ds.setup(stage='train', valid_split=[0.8, 0.2])  
    
    tra_ld = ds.train_dataloader(batch_size=cfger.batch_size, num_workers=cfger.num_workers)
    val_ld = ds.val_dataloader(batch_size=cfger.batch_size, num_workers=cfger.num_workers)
    num_cls = ds.dataset_info.num_classes
    
    # 3. prepare logger
    logger = get_wandb_logger(**cfger.wandb_args)
    logger.log_hyperparams(vars(cfger))
    
    res_net = get_backbone(cfger.backbone)
    clf = load_linear_clf(res_net, num_cls, cfger.ssl_method, cfger.ssl_args)
    
    tra_cfg = {'deterministic':tra_determin, 'logger':logger,
                'max_epochs':cfger.epochs, **cfger.compute_cfg}
    tra_cfg['strategy'] = DDPPlugin(find_unused_parameters=True)
    trainer = Trainer(sync_batchnorm=True, **tra_cfg)
    trainer.fit(clf, tra_ld)
    if 'ckpt_path' in cfger:
        trainer.save_checkpoint(cfger.ckpt_path)

    #trainer.test(val_ld, cfger.ckpt_path)
          

if __name__== "__main__":
    import os
    from easy_configer.Configer import Configer

    cfg_path = os.environ['CONFIGER_PATH']
    
    cfger = Configer("The configuration for pretrain phase of SSL.", cmd_args=True)
    cfger.cfg_from_ini(cfg_path)

    # 0. confirm repoducerbility
    if cfger.seed != -1:
        seed_everything(cfger.seed, workers=True)

    if 'pretrain' == cfg_path.split(os.sep)[-2]:
        pretrain_proc(cfger)
    else:
        linear_eval(cfger)