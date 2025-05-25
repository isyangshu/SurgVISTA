import logging
import torch
from torch import optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from torch import nn
from pycm import ConfusionMatrix
import numpy as np
import pickle
import csv
import torch.nn.functional as F

class FeatureExtraction(LightningModule):
    def __init__(self, hparams, model, dataset):
        super(FeatureExtraction, self).__init__()
        self.save_hyperparameters(hparams)
        self.batch_size = hparams.batch_size
        self.dataset = dataset
        self.ce_loss = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(self.dataset.class_weights)).float())
        # Cholec80
        # self.current_video_idx = int(list(self.dataset.df["test"].keys())[0][5:])
        # M2CAI16
        # self.current_video_idx = int(list(self.dataset.df["test"].keys())[0][6:])
        # AutoLaparo
        # self.current_video_idx = int(list(self.dataset.df["test"].keys())[0])
        # CATARACT/Cataract101/Cataract21
        self.current_video_idx = int(list(self.dataset.df["test"].keys())[0])

        self.init_metrics()

        # store model
        self.current_stems = []
        self.current_phase_labels = []
        self.current_p_phases = []
        self.len_test_data = len(self.dataset.data["test"])
        self.model = model
        self.best_metrics_high = {"val_acc_phase": 0}
        self.test_acc_per_video = {}
        self.pickle_path = None

    def init_metrics(self):
        self.train_acc_phase = pl.metrics.Accuracy()
        self.val_acc_phase = pl.metrics.Accuracy()
        self.test_acc_phase = pl.metrics.Accuracy()

    def set_export_pickle_path(self):
        self.pickle_path = self.hparams.output_path / "cataract101_pickle_export"
        self.pickle_path.mkdir(exist_ok=True)
        print(f"setting export_pickle_path: {self.pickle_path}")

    # ---------------------
    # TRAINING
    # ---------------------

    def forward(self,x):
        stem, phase = self.model.forward(x)
        return stem, phase

    def loss_phase(self, p_phase, labels_phase):
        loss_phase = self.ce_loss(p_phase, labels_phase.type(torch.int64))
        return loss_phase

    def training_step(self, batch, batch_idx):
        x, y_phase, batch_info = batch
        _, p_phase = self.forward(x)
        loss = self.loss_phase(p_phase, y_phase)
        # acc_phase, loss
        p_phase = F.softmax(p_phase, dim=1)
        self.train_acc_phase(p_phase, y_phase)
        self.log("train_acc_phase", self.train_acc_phase, on_epoch=True, on_step=True)
        self.log("loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_phase, batch_info = batch
        _, p_phase = self.forward(x)
        loss = self.loss_phase(p_phase, y_phase)
        # acc_phase, loss
        p_phase = F.softmax(p_phase, dim=1)
        self.val_acc_phase(p_phase, y_phase)
        self.log("val_acc_phase", self.val_acc_phase, on_epoch=True, on_step=False)
        self.log("val_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)

    def get_phase_acc(self, true_label, pred):
        pred = torch.FloatTensor(pred)
        pred_phase = torch.softmax(pred, dim=1)
        labels_pred = torch.argmax(pred_phase, dim=1).cpu().numpy()
        cm = ConfusionMatrix(
            actual_vector=true_label,
            predict_vector=labels_pred,
        )
        return cm.Overall_ACC, cm.PPV, cm.TPR, cm.classes, cm.F1_Macro

    def save_to_drive(self, vid_index):
        acc, ppv, tpr, keys, f1 = self.get_phase_acc(self.current_phase_labels, self.current_p_phases)
        save_path = self.pickle_path
        save_path.mkdir(exist_ok=True)
        save_path_txt = save_path / f"video_{vid_index:02d}_acc.txt"
        save_path_vid = save_path / f"video_{vid_index:02d}.pkl"
        save_label = self.pickle_path / "phase_annotations"
        save_pred = self.pickle_path / "prediction"
        save_label.mkdir(exist_ok=True)
        save_pred.mkdir(exist_ok=True)
        save_path_label = save_label / f"video-{str(vid_index).zfill(2)}.txt"
        save_path_pred = save_pred / f"video-{str(vid_index).zfill(2)}.txt"

        with open(save_path_txt, "w") as f:
            f.write(
                f"vid: {vid_index}; acc: {acc}; ppv: {ppv}; tpr: {tpr}; keys: {keys}; f1: {f1}"
            )
            self.test_acc_per_video[vid_index] = acc
            print(
                f"save video {vid_index} | acc: {acc:.4f} | f1: {f1}"
            )
        with open(save_path_vid, 'wb') as f:
            pickle.dump([
                np.asarray(self.current_stems),
                np.asarray(self.current_p_phases),
                np.asarray(self.current_phase_labels)
            ], f)

        assert len(self.current_phase_labels) == len(self.current_p_phases)
        with open(save_path_label, "w") as f:
            f.write("Frame")
            f.write("\t")
            f.write("Phase")
            f.write("\n")
            for i in range(len(self.current_phase_labels)):
                f.write(str(i))
                f.write("\t")
                f.write(str(self.current_phase_labels[i]))
                f.write("\n")
        with open(save_path_pred, "w") as f:
            f.write("Frame")
            f.write("\t")
            f.write("Phase")
            f.write("\n")
            for i in range(len(self.current_p_phases)):
                f.write(str(i))
                f.write("\t")
                f.write(str(np.argmax(self.current_p_phases[i])))
                f.write("\n")

    def test_step(self, batch, batch_idx):
        x, y_phase, (vid_idx, original_img_index, img_index) = batch
        vid_idx_raw = np.array([int(i) for i in vid_idx])
        with torch.no_grad():
            stem, y_hat = self.forward(x)
        y_hat_ = F.softmax(y_hat, dim=1)
        self.test_acc_phase(y_hat_, y_phase)
        #self.log("test_acc_phase", self.test_acc_phase, on_epoch=True, on_step=True)
        vid_idxs, indexes = np.unique(vid_idx_raw, return_index=True)
        vid_idxs = [int(x) for x in vid_idxs]
        index_next = len(vid_idx) if len(vid_idxs) == 1 else indexes[1]
        for i in range(len(vid_idxs)):
            vid_idx = vid_idxs[i]
            index = indexes[i]
            if int(vid_idx) != int(self.current_video_idx):
                self.save_to_drive(self.current_video_idx)
                self.current_stems = []
                self.current_phase_labels = []
                self.current_p_phases = []
                if len(vid_idxs) <= i + 1:
                    index_next = len(vid_idx_raw)
                else:
                    index_next = indexes[i+1]  # for the unlikely case that we have 3 phases in one batch
                self.current_video_idx = vid_idx
            y_hat_numpy = np.asarray(y_hat.cpu()).squeeze()
            self.current_p_phases.extend(
                np.asarray(y_hat_numpy[index:index_next, :]).tolist())
            self.current_stems.extend(
                stem[index:index_next, :].cpu().detach().numpy().tolist())
            y_phase_numpy = y_phase.cpu().numpy()
            self.current_phase_labels.extend(
                np.asarray(y_phase_numpy[index:index_next]).tolist())

        if (batch_idx + 1) * self.hparams.batch_size >= self.len_test_data:
            self.save_to_drive(vid_idx)
            print(f"Finished extracting all videos...")

    def test_epoch_end(self, outputs):
        print('Training ID', self.dataset.vids_for_training)
        print('Validation ID', self.dataset.vids_for_val)
        print('Test ID', self.dataset.vids_for_test)
        print('Total ID', self.test_acc_per_video.keys())
        if len(self.dataset.vids_for_training) + len(self.dataset.vids_for_val) != len(self.test_acc_per_video.keys()):
            self.dataset.vids_for_test = list(set(self.test_acc_per_video.keys()) - set(self.dataset.vids_for_training) - set(self.dataset.vids_for_val))
            print('New Test ID', self.dataset.vids_for_test)
        self.log("test_acc_train", np.mean(np.asarray([self.test_acc_per_video[x]for x in
                                                       self.dataset.vids_for_training])))
        self.log("test_acc_val", np.mean(np.asarray([self.test_acc_per_video[x]for x in
                                                     self.dataset.vids_for_val])))
        self.log("test_acc_test", np.mean(np.asarray([self.test_acc_per_video[x] for x in
                                                      self.dataset.vids_for_test])))
        self.log("test_acc", float(self.test_acc_phase.compute()))

        result_path = self.hparams.output_path / "result.csv"
        with open(result_path, mode='w') as file:
            writer = csv.writer(file)
            writer.writerow(["Metric", "Training", "Validation", "Test"])
            writer.writerow(["Accuracy", 
                     np.mean(np.asarray([self.test_acc_per_video[x] for x in self.dataset.vids_for_training])),
                     np.mean(np.asarray([self.test_acc_per_video[x] for x in self.dataset.vids_for_val])),
                     np.mean(np.asarray([self.test_acc_per_video[x] for x in self.dataset.vids_for_test]))])
            writer.writerow(["Image-level Accuracy", self.test_acc_phase.compute()])
    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        return whatever optimizers we want here
        :return: list of optimizers
        """
        optimizer = optim.Adam(self.parameters(),
                               lr=self.hparams.learning_rate)
        print(optimizer)
        #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer]  #, [scheduler]

    def __dataloader(self, split=None):
        dataset = self.dataset.data[split]
        if self.hparams.batch_size > self.hparams.model_specific_batch_size_max:
            print(
                f"The choosen batchsize is too large for this model."
                f" It got automatically reduced from: {self.hparams.batch_size} to {self.hparams.model_specific_batch_size_max}"
            )
            self.hparams.batch_size = self.hparams.model_specific_batch_size_max

        if split == "val" or split == "test":
            should_shuffle = False
        else:
            should_shuffle = True
        print(f"split: {split} - shuffle: {should_shuffle}")
        worker = self.hparams.num_workers
        if split == "test":
            print(
                "worker set to 0 due to test"
            )  # otherwise for extraction the order in which data is loaded is not sorted e.g. 1,2,3,4, --> 1,5,3,2
            worker = 0

        loader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            shuffle=should_shuffle,
            num_workers=worker,
            pin_memory=True,
        )
        return loader

    def train_dataloader(self):
        """
        Intialize train dataloader
        :return: train loader
        """
        dataloader = self.__dataloader(split="train")
        logging.info("training data loader called - size: {}".format(
            len(dataloader.dataset)))
        return dataloader

    def val_dataloader(self):
        """
        Initialize val loader
        :return: validation loader
        """
        dataloader = self.__dataloader(split="val")
        logging.info("validation data loader called - size: {}".format(
            len(dataloader.dataset)))
        return dataloader


    def test_dataloader(self):
        dataloader = self.__dataloader(split="test")
        logging.info("test data loader called  - size: {}".format(
            len(dataloader.dataset)))
        print(f"starting video idx for testing: {self.current_video_idx}")
        self.set_export_pickle_path()
        return dataloader

    @staticmethod
    def add_module_specific_args(parser):  # pragma: no cover
        cholec_fe_module = parser.add_argument_group(
            title='cholec_fe_module specific args options')
        cholec_fe_module.add_argument("--learning_rate",
                                      default=0.001,
                                      type=float)
        cholec_fe_module.add_argument("--optimizer_name",
                                      default="adam",
                                      type=str)
        cholec_fe_module.add_argument("--batch_size", default=32, type=int)
        return parser