# The backbone will directly call re-use the torch.hub
from torch import hub as tch_hub

def get_backbone(backbone_type=None): 
    if not backbone_type in tch_hub.list():
        raise ValueError(f"The torch hub only support {tch_hub.list()}")
    
    return getattr(tch_hub, backbone_type)
    
def wrap_ssl_method(feature_extractor):
    ...
