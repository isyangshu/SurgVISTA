from pathlib import Path
from torch.utils.data import Dataset
import pickle
import numpy as np
import pandas as pd


class M2CAI16Helper(Dataset):
    def __init__(self, hparams, data_p, dataset_split=None):
        assert dataset_split != None
        self.data_p = data_p
        assert hparams.data_root != ""
        self.data_root = Path(hparams.data_root)
        self.number_vids = len(self.data_p)
        self.dataset_split = dataset_split
        print(self.dataset_split, len(self.data_p))

    def __len__(self):
        return self.number_vids

    def __getitem__(self, index):
        vid_id = index
        p = self.data_root / self.data_p[vid_id]
        unpickled_x = pd.read_pickle(p)
        stem = np.asarray(unpickled_x[0],
                          dtype=np.float32)
        y_hat = np.asarray(unpickled_x[1],
                           dtype=np.float32)
        y = np.asarray(unpickled_x[2])
        return stem, y_hat, y, self.data_p[vid_id].split(".")[0]


class M2CAI16():
    def __init__(self, hparams):
        self.name = "M2CAI16Pickle"
        self.hparams = hparams
        self.class_labels = [
            "Preparation",
            "CalotTriangleDissection",
            "ClippingCutting",
            "GallbladderDissection",
            "GallbladderPackaging",
            "CleaningCoagulation",
            "GallbladderRetraction",
            "Other"
        ]
        self.out_features = self.hparams.out_features

        self.data_p = {}
        self.data_p["train"] = [(f"video_{i:02d}.pkl") for i in range(1, 28)]
        self.data_p["val"] = [(f"video_{i:02d}.pkl") for i in range(28, 42)]
        self.data_p["test"] = [(f"video_{i:02d}.pkl") for i in range(28, 42)]

        # Datasplit is equal to Endonet and Multi-Task Recurrent ConvNet with correlation loss for surgical vid analysis
        self.weights = {}
        self.weights["train"] = [1.0] * 8
        self.weights["train_log"] = [1.0] * 8

        self.data = {}
        for split in ["train", "val", "test"]:
            self.data[split] = M2CAI16Helper(hparams,
                                              self.data_p[split],
                                              dataset_split=split)

        print(
            f"train size: {len(self.data['train'])} - val size: {len(self.data['val'])} - test size:"
            f" {len(self.data['test'])}")

    @staticmethod
    def add_dataset_specific_args(parser):  # pragma: no cover
        m2cai16_specific_args = parser.add_argument_group(
            title='M2CAI16-workflow dataset specific args options')

        return parser
