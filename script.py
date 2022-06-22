from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DDPPlugin

from datamanager import get_datamanager
from model import get_backbone, load_ssl_method


def main(cfger):
    # 0. confirm repoducerbility
    tra_determin = False
    if cfger.seed != -1:
        seed_everything(cfger.seed, workers=True)
        tra_determin = True

    # 2. prepare SSL model
    res_net = get_backbone(cfger.backbone)
    ssl_model = load_ssl_method(res_net, 10, cfger.ssl_method, cfger.ssl_args)
    breakpoint()

if __name__ == "__main__":
    import os
    from easy_configer.Configer import Configer
    
    cfger = Configer("The configuration for pretrain phase of SSL.", cmd_args=True)
    cfger.cfg_from_ini( os.environ['CONFIGER_PATH'] )

    main(cfger)