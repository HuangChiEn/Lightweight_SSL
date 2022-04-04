import torch
import logging


def seed_everything(seed: int):
    if seed < 0:
        logging.info('[INFO] Disable seed')
        return

    ## setup the seed for 3rd (non-torch) op
    random.seed(seed)  
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)  

    ## setup the seed for torch op
    torch.manual_seed(seed)  # seed for cpu/gpu dev
    # setup the seed for cuda-algo (may decrease performance)
    torch.cuda.manual_seed(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logging.info("[INFO] Setting SEED: " + str(seed)) 


def print_info(feature_extractor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tot_params = sum(p.numel() for p in feature_extractor.parameters() if p.requires_grad)
    print(f"[INFO] {feature_extractor.name} loaded in memory.")
    print(f"[INFO] Feature size: {feature_extractor.feature_size}")
    print(f"[INFO] Feature extractor TOT trainable params: {tot_params}")
    if(torch.cuda.is_available() == False): 
        print("[WARNING] CUDA is not available.")
    else:
        print(f"[INFO] Found {torch.cuda.device_count()} GPU(s) available.")

    print(f"[INFO] Device type: {device}") 


def load_ckpt(model, checkpoint):
    # NOTE: the checkpoint must be loaded AFTER 
    # the model has been allocated into the device.
    if(checkpoint!=""):
        print("Loading checkpoint: " + str(checkpoint))
        model.load(checkpoint)
        print("Loading checkpoint: Done!")
    return model