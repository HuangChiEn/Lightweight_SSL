import pytorch_lightning as pl
from torch.optim import Adam, SGD, lr_scheduler
from model.solver.lr_scheduler import LARS
from torch import nn
from torch import optim
import torch.nn.functional as F
from util_tool.utils import accuracy_at_k

class Linear_classifier(pl.LightningModule):

    def __init__(self, backbone, in_feature, num_classes=1000):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.feature_extractor = backbone
        self.classifier = nn.Linear(in_feature, num_classes)

    def configure_optimizers(self):
        #optimizer = LARS(self.classifier.parameters(), lr=1.6, momentum=0.9, weight_decay=0)
        #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=90)
        #return [optimizer], [scheduler]
        optimizer = Adam(self.classifier.parameters(), lr=2e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=90)
        return [optimizer], [scheduler]
        

    def forward(self, X, targets):
        feats = self.feature_extractor(X).detach()
        logits = self.classifier(feats)
        return logits

    def training_step(self, batch, batch_idx):
        X, targets = batch
        logits = self(X[0], targets)  # plz setup n_arg_crop = [1] to perform only one view

        top_k_max = min(5, logits.size(1))
        acc1, acc5 = accuracy_at_k(logits, targets, top_k=(1, top_k_max))
        loss = F.cross_entropy(logits, targets, ignore_index=-1)

        metrics = { 
            "lin_loss": loss,
            "lin_acc1": acc1,
            "lin_acc5": acc5,
        }
        self.log_dict(metrics, on_step=True, on_epoch=False, sync_dist=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        X, targets = batch
        logits = self(X[0], targets)  # plz setup n_arg_crop = [1] to perform only one view

        top_k_max = min(5, logits.size(1))
        test_acc, _ = accuracy_at_k(logits, targets, top_k=(1, top_k_max))

        self.log_dict({'test_acc': test_acc})


    ## Progressbar adjustment of output console
    def on_epoch_start(self):
        print('\n')

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        tqdm_dict.pop("loss", None)
        return tqdm_dict