# The backbone will directly call re-use the torch.hub
from torch import nn
import torchvision.models as model_hub
from model.rnnclvr import RNN_CLVR
#from model.simclr import Sim_CLR

SSL_METHODS = {
    'rnnclvr' : RNN_CLVR,
    #'simclr' : Sim_CLR
}

def get_backbone(backbone_type): 
    
    def remove_fc_head(backbone):
        backbone.fc = nn.Identity() 
        return backbone

    mod_catelog = lst = [ mod for mod in dir(model_hub) if not mod.startswith('_') and not mod[0].isupper() ]
    if not backbone_type in mod_catelog:
        raise ValueError(f"The torch hub only support {mod_catelog}")
    
    raw_backbone = getattr(model_hub, backbone_type)(pretrained=False)  # disable num_classes=avg_pool_dim flag
    return remove_fc_head(raw_backbone)
    

def wrap_ssl_method(backbone, num_cls, ssl_method, ssl_args):
    if not ssl_method in SSL_METHODS:
        raise ValueError(f"The ssl method {ssl_method} are not supported yet, we only support {SSL_METHODS.keys()} currently..")
    
    ssl_args.update( {'num_of_cls':num_cls} )
    return SSL_METHODS[ssl_method](backbone=backbone, **ssl_args)