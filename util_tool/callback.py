from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

def get_model_ckpt(callbk_lst, dirpath, filename, every_n_epochs, save_top_k, monitor, **_):
    checkpoint_callback = ModelCheckpoint(dirpath=dirpath, filename=filename, every_n_epochs=every_n_epochs,
                                            save_top_k=save_top_k, monitor=monitor) 
    callbk_lst.append(checkpoint_callback)

def get_lr_monitor(callbk_lst, logging_interval='step'):
    lr_monitor_callback = LearningRateMonitor(logging_interval=logging_interval)
    callbk_lst.append(lr_monitor_callback)
    