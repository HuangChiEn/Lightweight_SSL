from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DDPPlugin

from model import load_linear_clf, get_backbone
from datamanager import get_datamanager
from util_tool.utils import get_wandb_logger


# finally, i'll merge into single script
def main(cfger):
    # 0. confirm repoducerbility
    tra_determin = False
    if cfger.seed != -1:
        seed_everything(cfger.seed, workers=True)
        tra_determin = True

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
    #trainer.test(clf, val_ld)
          

if __name__== "__main__":
    import os
    from easy_configer.Configer import Configer
    
    cfger = Configer("The configuration for pretrain phase of SSL.", cmd_args=True)
    cfger.cfg_from_ini( os.environ['CONFIGER_PATH'] )

    main(cfger)