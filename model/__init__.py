# The backbone will directly call re-use the torch.hub
from torch import nn
import torchvision.models as model_hub
from model.rnnclvr import RNN_CLVR
#from model.simclr import Sim_CLR

SSL_METHODS = {
    'rnnclvr' : RNN_CLVR,
    #'simclr' : Sim_CLR
}

def get_backbone(backbone_type=None, out_dim=128): 
    
    def modified_head(raw_mod):
        dim_mlp = raw_mod.fc.in_features
        # add mlp projection head
        raw_mod.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), raw_mod.fc)
        return raw_mod

    mod_catelog = lst = [ mod for mod in dir(model_hub) if not mod.startswith('_') and not mod[0].isupper() ]
    if not backbone_type in mod_catelog:
        raise ValueError(f"The torch hub only support {mod_catelog}")
    
    raw_mod = getattr(model_hub, backbone_type)(pretrained=False, num_classes=out_dim)
    return modified_head(raw_mod)
    

def wrap_ssl_method(ssl_method, feature_extractor, **kwargs):
    if not ssl_method in SSL_METHODS:
        raise ValueError(f"The ssl method {ssl_method} are not supported yet, we only support {SSL_METHODS.keys()} currently..")
    
    return SSL_METHODS[ssl_method](feature_extractor=feature_extractor, **kwargs)
