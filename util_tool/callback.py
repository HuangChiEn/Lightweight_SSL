from pytorch_lightning.callbacks import ModelCheckpoint

def get_model_ckpt(callbk_lst, dirpath, filename, every_n_epochs, save_top_k, monitor):
    checkpoint_callback = ModelCheckpoint(dirpath=dirpath, filename=filename, every_n_epochs=every_n_epochs,
                                            save_top_k=save_top_k, monitor=monitor)
    callbk_lst.append(checkpoint_callback)

