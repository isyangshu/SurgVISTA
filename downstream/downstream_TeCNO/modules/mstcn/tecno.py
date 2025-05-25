import logging
import torch
from torch import optim
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from pytorch_lightning.metrics.classification.helpers import _input_format_classification
from pytorch_lightning.core.lightning import LightningModule
from utils_tecno.metric_helper import AccuracyStages, RecallOverClasse, PrecisionOverClasses
from torch import nn
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
import pickle
import csv

class TeCNO(LightningModule):
    def __init__(self, hparams, model, dataset):
        super(TeCNO, self).__init__()
        self.hparams = hparams
        self.batch_size = hparams.batch_size
        self.dataset = dataset
        self.model = model
        self.weights_train = np.asarray(self.dataset.weights["train"])
        self.ce_loss = nn.CrossEntropyLoss(weight=torch.from_numpy(self.weights_train).float())
        self.set_export_pickle_path()

    def set_export_pickle_path(self):
        self.hparams.output_path.mkdir(exist_ok=True)
        self.pickle_path = self.hparams.output_path / "cataract_pickle_export"
        self.pickle_path.mkdir(exist_ok=True)
        print(f"setting export_pickle_path: {self.pickle_path}")

    def get_phase_acc(self, true_label, pred):
        pred = torch.FloatTensor(pred)
        pred_phase = torch.softmax(pred, dim=1)
        labels_pred = torch.argmax(pred_phase, dim=1).cpu().numpy()
        true_label = true_label.cpu().numpy()
        acc = np.mean(labels_pred == true_label)
        return acc
    
    def forward(self, x):
        video_fe = x.transpose(2, 1)
        y_classes = self.model.forward(video_fe)
        y_classes = torch.softmax(y_classes, dim=2)
        return y_classes

    def loss_function(self, y_classes, labels):
        stages = y_classes.shape[0]
        clc_loss = 0
        for j in range(stages):  ### make the interuption free stronge the more layers.
            p_classes = y_classes[j].squeeze().transpose(1, 0)
            ce_loss = self.ce_loss(p_classes, labels.squeeze())
            clc_loss += ce_loss
        clc_loss = clc_loss / (stages * 1.0)
        return clc_loss

    def get_class_acc(self, y_true, y_classes):
        y_true = y_true.squeeze()
        y_classes = y_classes.squeeze()
        y_classes = torch.argmax(y_classes, dim=0)
        acc_classes = torch.sum(
            y_true == y_classes).float() / (y_true.shape[0] * 1.0)
        return acc_classes

    def training_step(self, batch, batch_idx):
        stem, y_hat, y_true, vid_idx = batch
        y_pred = self.forward(stem)
        loss = self.loss_function(y_pred, y_true)
        self.log("loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        y_pred = y_pred[-1].cpu()
        acc = self.get_phase_acc(y_true, y_pred)
        self.log("train_acc_phase", acc, on_epoch=True, on_step=False)
        return {"loss":loss, "acc": acc}
    
    def training_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        :return:
        self.max_acc_last_stage = {"epoch": 0, "acc": 0}
        self.max_acc_global = {"epoch": 0, "acc": 0 , "stage": 0}
        """
        train_acc_video = torch.stack([torch.tensor(o["acc"]) if isinstance(o["acc"], float) else o["acc"] for o in outputs]).mean()
        self.log("train_acc_video", train_acc_video)
        print(train_acc_video)

    def validation_step(self, batch, batch_idx):
        stem, y_hat, y_true, vid_idx = batch
        y_pred = self.forward(stem)
        val_loss = self.loss_function(y_pred, y_true)
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True, on_step=False)
        y_pred = y_pred[-1].cpu()
        acc = self.get_phase_acc(y_true, y_pred)
        self.log("val_acc_phase", acc, on_epoch=True, on_step=False)
        return {"val_loss": val_loss, "acc": acc}

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        :return:
        self.max_acc_last_stage = {"epoch": 0, "acc": 0}
        self.max_acc_global = {"epoch": 0, "acc": 0 , "stage": 0}
        """
        val_acc_video = torch.stack([torch.tensor(o["acc"]) if isinstance(o["acc"], float) else o["acc"] for o in outputs]).mean()
        self.log("val_acc_video", val_acc_video)

    def save_to_drive(self, vid_index, y_pred, y_true, acc):
        save_path = self.pickle_path
        save_path.mkdir(exist_ok=True)
        save_path_txt = save_path / (vid_index[0] + "_acc.txt")
        save_label = self.pickle_path / "phase_annotations"
        save_pred = self.pickle_path / "prediction"
        save_label.mkdir(exist_ok=True)
        save_pred.mkdir(exist_ok=True)
        save_path_label = save_label / (vid_index[0].replace('_', '-') + ".txt")
        save_path_pred = save_pred / (vid_index[0].replace('_', '-') + ".txt")

        with open(save_path_txt, "w") as f:
            f.write(
                f"vid: {vid_index}; acc: {acc}; "
            )

        y_pred = torch.softmax(y_pred, dim=1)
        y_pred = torch.argmax(y_pred, dim=1).squeeze().cpu().numpy()
        y_true = y_true.squeeze().cpu().numpy()
        with open(save_path_label, "w") as f:
            f.write("Frame")
            f.write("\t")
            f.write("Phase")
            f.write("\n")
            for i in range(len(y_true)):
                f.write(str(i))
                f.write("\t")
                f.write(str(y_true[i]))
                f.write("\n")
        with open(save_path_pred, "w") as f:
            f.write("Frame")
            f.write("\t")
            f.write("Phase")
            f.write("\n")
            for i in range(len(y_pred)):
                f.write(str(i))
                f.write("\t")
                f.write(str(y_pred[i]))
                f.write("\n")

    def test_step(self, batch, batch_idx):
        stem, y_hat, y_true, vid_idx = batch
        y_pred = self.forward(stem)
        val_loss = self.loss_function(y_pred, y_true)
        self.log("test_loss", val_loss, on_epoch=True, prog_bar=True, on_step=False)
        y_pred = y_pred[-1].cpu()
        acc = self.get_phase_acc(y_true, y_pred)
        self.log("test_acc_phase", acc, on_epoch=True, on_step=False)
        self.save_to_drive(vid_idx, y_pred, y_true, acc)
        return {"loss": val_loss, "acc": acc}

    def test_epoch_end(self, outputs):
        test_acc = torch.stack([torch.tensor(o["acc"]) if isinstance(o["acc"], float) else o["acc"] for o in outputs]).mean()
        self.log("test_acc_video", test_acc)
        result_path = self.hparams.output_path / "result.csv"
        with open(result_path, mode='w') as file:
            writer = csv.writer(file)
            writer.writerow(["Metric", "Test"])
            writer.writerow(["Accuracy", test_acc])

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(),
                               lr=self.hparams.learning_rate)
        return [optimizer]  #, [scheduler]

    def __dataloader(self, split=None):
        dataset = self.dataset.data[split]
        should_shuffle = False
        if split == "train":
            should_shuffle = True
        # when using multi-node (ddp) we need to add the  datasampler
        train_sampler = None
        # if self.use_ddp:
        #     train_sampler = DistributedSampler(dataset)
        #     should_shuffle = False
        print(f"split: {split} - shuffle: {should_shuffle}")
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            shuffle=should_shuffle,
            sampler=train_sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )
        return loader

    def train_dataloader(self):
        dataloader = self.__dataloader(split="train")
        logging.info("training data loader called - size: {}".format(
            len(dataloader.dataset)))
        return dataloader

    def val_dataloader(self):
        dataloader = self.__dataloader(split="val")
        logging.info("validation data loader called - size: {}".format(
            len(dataloader.dataset)))
        return dataloader

    def test_dataloader(self):
        dataloader = self.__dataloader(split="test")
        logging.info("test data loader called  - size: {}".format(
            len(dataloader.dataset)))
        return dataloader

    @staticmethod
    def add_module_specific_args(parser):  # pragma: no cover
        regressiontcn = parser.add_argument_group(
            title='regression tcn specific args options')
        regressiontcn.add_argument("--learning_rate",
                                   default=0.001,
                                   type=float)
        regressiontcn.add_argument("--optimizer_name",
                                   default="adam",
                                   type=str)
        regressiontcn.add_argument("--batch_size", default=1, type=int)

        return parser
