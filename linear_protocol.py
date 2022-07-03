from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DDPPlugin

from model import load_linear_clf, get_backbone, wrap_ssl_method
from datamanager import get_pretrain_datamanager, get_finetune_datamanager
from util_tool.utils import get_wandb_logger
from util_tool.callback import get_model_ckpt, get_lr_monitor


def pretrain_proc(cfger, DEFAULT_DATA_DIR='/data/tch_ds'):
    '''
        There're some function you can customized to enable it :
        1. you can split training set by the given ratio in prepare dataset phase,
            if getattr(cfger, 'val_split', None):
                ds.setup(stage='train', valid_split=cfger.val_split)  
        2. you can manuel saving ckpt, but the model_ckpt callbk already well-structured..
            if getattr(cfger, 'ckpt_path', None):
                trainer.save_checkpoint(cfger.ckpt_path)
    '''
    # 1. prepare dataset
    data_dir = cfger.data_dir if getattr(cfger, 'data_dir', None) else DEFAULT_DATA_DIR
    ds = get_pretrain_datamanager(cfger.dataset, data_dir, cfger.aug_crop_lst)
    ds.setup(stage='train')

    tra_ld = ds.train_dataloader(batch_size=cfger.batch_size, num_workers=cfger.num_workers)
    num_cls = ds.dataset_info.num_classes

    # 2. prepare SSL model
    res_net = get_backbone(cfger.backbone)
    ssl_model = wrap_ssl_method(res_net, cfger.ssl_method, cfger.ssl_args, num_cls, cfger.epochs)

    # 3. prepare logger
    logger = get_wandb_logger(**cfger.wandb_args)
    logger.log_hyperparams(vars(cfger))   # log all cfg, it's useful to visual in wandb
    
    # 4.prepare callback : model_ckpt, lr_monitor
    callbk_lst = []
    get_model_ckpt(callbk_lst, **cfger.ckpt_args)
    get_lr_monitor(callbk_lst)

    # 5. prepare trainer config & conduct training loop
    tra_cfg = {'deterministic':bool(cfger.seed != -1), 'logger':logger, 'callbacks':callbk_lst, 
                'max_epochs':cfger.epochs, **cfger.compute_cfg}
    # setup 16-precision to reduce the GPU memory & 3xSpeedup training with slightly decrease of preformance
    tra_cfg['precision'] = 16 if tra_cfg['accelerator'] == 'gpu' else 32
    # setup find_unused_parameters=False to disable warning & speed-up
    tra_cfg['strategy'] = DDPPlugin(find_unused_parameters=False) if tra_cfg['strategy'] == 'ddp' else tra_cfg['strategy']

    trainer = Trainer(sync_batchnorm=True, **tra_cfg)  # sync_bn is necessary in multi-gpu
    trainer.fit(ssl_model, tra_ld)    


def linear_eval(cfger, only_eval):
    # 1. prepare dataset
    data_dir = cfger.data_dir if getattr(cfger, 'data_dir', None) else '/data'
    ds = get_finetune_datamanager(cfger.dataset, data_dir, cfger.aug_crop_lst)

    # 2. prepare linear classifier
    num_cls = ds.dataset_info.num_classes
    res_net = get_backbone(cfger.backbone)
    clf = load_linear_clf(res_net, num_cls, cfger.ssl_method, cfger.ssl_args, cfger.pretrain_ckpt_path)
    
    if not only_eval:
        # 3. setup train loader
        ds.setup(stage='train')   # HACKME : valid_split=[0.8, 0.2], and built valid_step
        tra_ld = ds.train_dataloader(batch_size=cfger.batch_size, num_workers=cfger.num_workers)
        
        # 4. prepare logger
        logger = get_wandb_logger(**cfger.wandb_args)
        logger.log_hyperparams(vars(cfger))

        # 5. model_ckpt
        callbk_lst = []
        get_model_ckpt(callbk_lst, **cfger.ckpt_args)
        get_lr_monitor(callbk_lst)
    
        tra_cfg = {'deterministic':bool(cfger.seed != -1), 'logger':logger, 'callbacks':callbk_lst, 
                    'max_epochs':cfger.epochs, **cfger.compute_cfg}
        tra_cfg['strategy'] = DDPPlugin(find_unused_parameters=True)
        trainer = Trainer(sync_batchnorm=True, **tra_cfg)
        trainer.fit(clf, tra_ld)

    # 6. setup test loader
    ds.setup(stage='test')  
    tst_ld = ds.test_dataloader(batch_size=cfger.batch_size, num_workers=cfger.num_workers, shuffle=False)

    # load the previous ckpt or get the best ckpt according to the 'fit' results..
    ckpt_path = cfger.ckpt_args['ckpt_path'] if only_eval else callbk_lst[0].best_model_path
    test_args = {'model':clf, 'dataloaders':tst_ld, 'ckpt_path':ckpt_path}
    test_cfg = {'deterministic':bool(cfger.seed != -1), 'devices':1, 'accelerator':'gpu'}

    trainer = Trainer(**test_cfg)  
    trainer.test(**test_arg)


if __name__== "__main__":
    import os
    from easy_configer.Configer import Configer

    cfg_path = os.environ['CONFIGER_PATH']
    
    cfger = Configer("The configuration for pretrain phase of SSL.", cmd_args=True)
    cfger.cfg_from_ini(cfg_path)

    # 0. confirm repoducerbility
    if cfger.seed != -1:
        seed_everything(cfger.seed, workers=True)
        
    # trace cfg_path to dispatch the corresponding assignment..
    if 'pretrain' == cfg_path.split(os.sep)[-2]:
        pretrain_proc(cfger)
    else:
        if (cfger.ckpt_args['only_eval']) and (not 'ckpt_path' in cfger.ckpt_args):
            raise RuntimeError("The ckpt_path for loading trained linear classifier should be given, while the only_eval set to True!")
        linear_eval(cfger, only_eval=cfger.ckpt_args['only_eval'])