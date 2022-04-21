from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer

from datamanager import get_datamanager
from model import get_backbone, wrap_ssl_method

from util_tool.utils import get_wandb_logger
from util_tool.callback import get_model_ckpt

## HACKME : add auto_resumer
def main(cfger):
    # 0. confirm repoducerbility
    tra_determin = False
    if cfger.seed != -1:
        seed_everything(cfger.seed, workers=True)
        tra_determin = True

    # 1. prepare dataset
    ds = get_datamanager(cfger.dataset, cfger.aug_crop_lst)
    ds.setup(stage='train')  # support valid_split=[0.8, 0.2]
    tra_ld = ds.train_dataloader(batch_size=cfger.batch_size, num_workers=cfger.num_workers)
    num_cls = ds.dataset_info.num_classes

    # 2. prepare SSL model
    res_net = get_backbone(cfger.backbone)
    ssl_model = wrap_ssl_method(res_net, num_cls, cfger.ssl_method, cfger.ssl_args)

    # 3. prepare logger
    logger = get_wandb_logger(**cfger.wandb_args)
    logger.log_hyperparams(vars(cfger))
    
    # 4.prepare callback
    callbk_lst = []
    get_model_ckpt(callbk_lst, **cfger.ckpt_args)

    # 4. prepare trainer config from prev def & conduct training loop
    tra_cfg = {'deterministic':tra_determin, 'logger':logger, 'callbacks':callbk_lst, **cfger.compute_cfg}
    trainer = Trainer(**tra_cfg)
    trainer.fit(ssl_model, tra_ld)


if __name__ == "__main__":
    import os
    from easy_configer.Configer import Configer
    
    cfger = Configer("The configuration for pretrain phase of SSL.", cmd_args=True)
    cfger.cfg_from_ini( os.environ['CONFIGER_PATH'] )

    main(cfger)