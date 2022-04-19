from pytorch_lightning import seed_everything
from datamanager import get_datamanager
from model import get_backbone, wrap_ssl_method
from pytorch_lightning import Trainer

## HACKME : add auto_resumer
def main(cfger):
    # 0. confirm repoducerbility
    tra_determin = False
    if cfger.seed != -1:
        seed_everything(cfger.seed, workers=True)
        tra_determin = True

    # 1. prepare dataset
    ds = get_datamanager(cfger.dataset, cfger.aug_crop_lst)
    ds.setup(stage='train', valid_split=[0.8, 0.2])
    tra_ld = ds.train_dataloader(batch_size=cfger.batch_size, num_workers=cfger.num_workers)
    val_ld = ds.val_dataloader(batch_size=cfger.batch_size, num_workers=cfger.num_workers)

    # 2. prepare SSL model
    res_net = get_backbone(cfger.backbone)
    ssl_model = wrap_ssl_method(cfger.ssl_method, res_net)

    # 3. prepare trainer config & conduct training loop
    tra_cfg = {'deterministic':tra_determin, **cfger.compute_cfg}
    trainer = Trainer(**tra_cfg)
    trainer.fit(ssl_model, tra_ld, val_ld)


if __name__ == "__main__":
    import os
    import logging
    from easy_configer.Configer import Configer
    
    cfger = Configer("The configuration for pretrain phase of SSL.", cmd_args=True)
    cfger.cfg_from_ini( os.environ['CONFIGER_PATH'] )

    # loggin level setup
    logging.basicConfig(level=cfger.log_level)

    main(cfger)