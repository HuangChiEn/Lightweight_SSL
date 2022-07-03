# The backbone will directly call re-use the torch.hub,
#   we have plane to replce it into timm repo:https://github.com/rwightman/pytorch-image-models
#   or the vit repo:pip install vit-pytorch
from torch import nn
import torchvision.models as model_hub

from model.nnclr import NN_CLR
from model.simclr import Sim_CLR
from model.simsiam import Sim_Siam
from model.barlow_twin import Barlow_Twin
#from model.byol import BYOL

from model.linear_classifier import Linear_classifier

SSL_METHODS = {
    'nnclr' : NN_CLR,
    'simclr' : Sim_CLR,
    'simsiam' : Sim_Siam,
    'barlowtwin' : Barlow_Twin,
    #'byol' : BYOL
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
    

def wrap_ssl_method(backbone, ssl_method, ssl_args, num_cls, epochs):
    if not ssl_method in SSL_METHODS:
        raise ValueError(f"The ssl method {ssl_method} are not supported yet, we only support {SSL_METHODS.keys()} currently..")
    
    ssl_args.update( {'num_of_cls':num_cls, 'tot_epochs':epochs} )
    return SSL_METHODS[ssl_method](backbone=backbone, **ssl_args)


def load_linear_clf(backbone, num_cls, ssl_method, ssl_args, ckpt_path=None):
    
    def get_ssl_backbone(ckpt_path):
        other_args = {'backbone':backbone, 'num_of_cls':num_cls, **ssl_args}
        ssl_model = SSL_METHODS[ssl_method].load_from_checkpoint(ckpt_path, **other_args)
        return ssl_model.backbone, ssl_model.classifier.in_features
    
    backbone, in_features = get_ssl_backbone(ckpt_path)
    return Linear_classifier(backbone, in_features)


