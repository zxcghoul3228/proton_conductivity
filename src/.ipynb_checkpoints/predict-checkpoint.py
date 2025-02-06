import sys
import os
import copy
import warnings
from pathlib import Path
import shutil

import pytorch_lightning as pl

from moftransformer.config import ex
from moftransformer.config import config as _config
from moftransformer.utils.validation import (
    get_valid_config,
    get_num_devices,
    ConfigurationError,
)

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)

import os
import random
import json
import pickle

import numpy as np

import torch
from torch.nn.functional import interpolate


# for regression task
N_ADD_FEATURES = 4


# Modified dataset
class Dataset1(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str,
        nbr_fea_len: int,
        draw_false_grid=True,
        downstream="",
        tasks=[],
    ):
        """
        Dataset for pretrained MOF.
        Args:
            data_dir (str): where dataset cif files and energy grid file; exist via model.utils.prepare_data.py
            split(str) : train, test, split
            draw_false_grid (int, optional):  how many generating false_grid_data
            nbr_fea_len (int) : nbr_fea_len for gaussian expansion
        """
        super().__init__()
        self.data_dir = data_dir
        self.draw_false_grid = draw_false_grid
        self.split = split

        assert split in {"train", "test", "val"}
        if downstream:
            path_file = os.path.join(data_dir, f"{split}_{downstream}.json")
            path_file_features = os.path.join(
                data_dir, f"{split}_{downstream}_features.json"
            )  # path to file with additional (global) features
        else:
            path_file = os.path.join(data_dir, f"{split}.json")
            path_file_features = os.path.join(
                data_dir, f"{split}_features.json"
            )  # path to file with additional features
        print(f"read {path_file}...")
        print(f"read {path_file_features}...")

        if not os.path.isfile(path_file):
            raise FileNotFoundError(
                f"{path_file} doesn't exist. Check 'root_dataset' in config"
            )
        # if not os.path.isfile(path_file_features):
        # self.with_add_features = False #flag for presence of add features
        # print('file with additional features does not exist')
        # else:
        # self.with_add_features = True #flag for presence of add features

        dict_target = json.load(open(path_file, "r"))
        self.cif_ids, self.targets = zip(*dict_target.items())

        dict_add_features = json.load(
            open(path_file_features, "r")
        )  # dict with add features
        self.add_features = [dict_add_features[cif_id] for cif_id in self.cif_ids]

        self.nbr_fea_len = nbr_fea_len

        self.tasks = {}

        for task in tasks:
            if task in ["mtp", "vfp", "moc", "bbc"]:
                path_file = os.path.join(data_dir, f"{split}_{task}.json")
                print(f"read {path_file}...")
                assert os.path.isfile(
                    path_file
                ), f"{path_file} doesn't exist in {data_dir}"

                dict_task = json.load(open(path_file, "r"))
                cif_ids, t = zip(*dict_task.items())
                self.tasks[task] = list(t)
                assert self.cif_ids == cif_ids, print(
                    "order of keys is different in the json file"
                )

    def __len__(self):
        return len(self.cif_ids)

    @staticmethod
    def make_grid_data(grid_data, emin=-5000.0, emax=5000, bins=101):
        """
        make grid_data within range (emin, emax) and
        make bins with logit function
        and digitize (0, bins)
        ****
            caution : 'zero' should be padding !!
            when you change bins, heads.MPP_heads should be changed
        ****
        """
        grid_data[grid_data <= emin] = emin
        grid_data[grid_data > emax] = emax

        x = np.linspace(emin, emax, bins)
        new_grid_data = np.digitize(grid_data, x) + 1

        return new_grid_data

    @staticmethod
    def calculate_volume(a, b, c, angle_a, angle_b, angle_c):
        a_ = np.cos(angle_a * np.pi / 180)
        b_ = np.cos(angle_b * np.pi / 180)
        c_ = np.cos(angle_c * np.pi / 180)

        v = a * b * c * np.sqrt(1 - a_**2 - b_**2 - c_**2 + 2 * a_ * b_ * c_)

        return v.item() / (60 * 60 * 60)  # normalized volume

    def get_raw_grid_data(self, cif_id):
        file_grid = os.path.join(self.data_dir, self.split, f"{cif_id}.grid")
        file_griddata = os.path.join(self.data_dir, self.split, f"{cif_id}.griddata16")

        # get grid
        with open(file_grid, "r") as f:
            lines = f.readlines()
            a, b, c = [float(i) for i in lines[0].split()[1:]]
            angle_a, angle_b, angle_c = [float(i) for i in lines[1].split()[1:]]
            cell = [int(i) for i in lines[2].split()[1:]]

        volume = self.calculate_volume(a, b, c, angle_a, angle_b, angle_c)

        # get grid data
        grid_data = pickle.load(open(file_griddata, "rb"))
        grid_data = self.make_grid_data(grid_data)
        grid_data = torch.FloatTensor(grid_data)

        return cell, volume, grid_data

    def get_grid_data(self, cif_id, draw_false_grid=False):
        cell, volume, grid_data = self.get_raw_grid_data(cif_id)
        ret = {
            "cell": cell,
            "volume": volume,
            "grid_data": grid_data,
        }

        if draw_false_grid:
            random_index = random.randint(0, len(self.cif_ids) - 1)
            cif_id = self.cif_ids[random_index]
            cell, volume, grid_data = self.get_raw_grid_data(cif_id)
            ret.update(
                {
                    "false_cell": cell,
                    "fale_volume": volume,
                    "false_grid_data": grid_data,
                }
            )
        return ret

    @staticmethod
    def get_gaussian_distance(distances, num_step, dmax, dmin=0, var=0.2):
        """
        Expands the distance by Gaussian basis
        (https://github.com/txie-93/cgcnn.git)
        """

        assert dmin < dmax
        _filter = np.linspace(
            dmin, dmax, num_step
        )  # = np.arange(dmin, dmax + step, step) with step = 0.2

        return np.exp(-((distances[..., np.newaxis] - _filter) ** 2) / var**2).float()

    def get_graph(self, cif_id):
        file_graph = os.path.join(self.data_dir, self.split, f"{cif_id}.graphdata")

        graphdata = pickle.load(open(file_graph, "rb"))
        # graphdata = ["cif_id", "atom_num", "nbr_idx", "nbr_dist", "uni_idx", "uni_count"]
        atom_num = torch.LongTensor(graphdata[1].copy())
        nbr_idx = torch.LongTensor(graphdata[2].copy()).view(len(atom_num), -1)
        nbr_dist = torch.FloatTensor(graphdata[3].copy()).view(len(atom_num), -1)

        nbr_fea = torch.FloatTensor(
            self.get_gaussian_distance(nbr_dist, num_step=self.nbr_fea_len, dmax=8)
        )

        uni_idx = graphdata[4]
        uni_count = graphdata[5]

        return {
            "atom_num": atom_num,
            "nbr_idx": nbr_idx,
            "nbr_fea": nbr_fea,
            "uni_idx": uni_idx,
            "uni_count": uni_count,
        }

    def get_tasks(self, index):
        ret = dict()
        for task, value in self.tasks.items():
            ret.update({task: value[index]})

        return ret

    def __getitem__(self, index):
        ret = dict()
        cif_id = self.cif_ids[index]
        target = self.targets[index]
        add_features = self.add_features[index]  # add features list[float]

        ret.update(
            {
                "cif_id": cif_id,
                "target": target,
                "add_features": add_features,  # add features list[float]
            }
        )
        ret.update(self.get_grid_data(cif_id, draw_false_grid=self.draw_false_grid))
        ret.update(self.get_graph(cif_id))

        ret.update(self.get_tasks(index))

        return ret

    @staticmethod
    def collate(batch, img_size):
        """
        collate batch
        Args:
            batch (dict): [cif_id, atom_num, nbr_idx, nbr_fea, uni_idx, uni_count,
                            grid_data, cell, (false_grid_data, false_cell), target]
            img_size (int): maximum length of img size

        Returns:
            dict_batch (dict): [cif_id, atom_num, nbr_idx, nbr_fea, crystal_atom_idx,
                                uni_idx, uni_count, grid, false_grid_data, target]
        """
        batch_size = len(batch)
        keys = set([key for b in batch for key in b.keys()])

        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        # graph
        batch_atom_num = dict_batch["atom_num"]
        batch_nbr_idx = dict_batch["nbr_idx"]
        batch_nbr_fea = dict_batch["nbr_fea"]

        crystal_atom_idx = []
        base_idx = 0
        for i, nbr_idx in enumerate(batch_nbr_idx):
            n_i = nbr_idx.shape[0]
            crystal_atom_idx.append(torch.arange(n_i) + base_idx)
            nbr_idx += base_idx
            base_idx += n_i

        dict_batch["atom_num"] = torch.cat(batch_atom_num, dim=0)
        dict_batch["nbr_idx"] = torch.cat(batch_nbr_idx, dim=0)
        dict_batch["nbr_fea"] = torch.cat(batch_nbr_fea, dim=0)
        dict_batch["crystal_atom_idx"] = crystal_atom_idx

        # grid
        batch_grid_data = dict_batch["grid_data"]
        batch_cell = dict_batch["cell"]
        new_grids = []

        for bi in range(batch_size):
            orig = batch_grid_data[bi].view(batch_cell[bi][::-1]).transpose(0, 2)
            if batch_cell[bi] == [30, 30, 30]:  # version >= 1.1.2
                orig = orig[None, None, :, :, :]
            else:
                orig = interpolate(
                    orig[None, None, :, :, :],
                    size=[img_size, img_size, img_size],
                    mode="trilinear",
                    align_corners=True,
                )
            new_grids.append(orig)
        new_grids = torch.concat(new_grids, axis=0)
        dict_batch["grid"] = new_grids

        if "false_grid_data" in dict_batch.keys():
            batch_false_grid_data = dict_batch["false_grid_data"]
            batch_false_cell = dict_batch["false_cell"]
            new_false_grids = []
            for bi in range(batch_size):
                orig = batch_false_grid_data[bi].view(batch_false_cell[bi])
                if batch_cell[bi] == [30, 30, 30]:  # version >= 1.1.2
                    orig = orig[None, None, :, :, :]
                else:
                    orig = interpolate(
                        orig[None, None, :, :, :],
                        size=[img_size, img_size, img_size],
                        mode="trilinear",
                        align_corners=True,
                    )
                new_false_grids.append(orig)
            new_false_grids = torch.concat(new_false_grids, axis=0)
            dict_batch["false_grid"] = new_false_grids

        dict_batch.pop("grid_data", None)
        dict_batch.pop("false_grid_data", None)
        dict_batch.pop("cell", None)
        dict_batch.pop("false_cell", None)

        return dict_batch


# MOFTransformer version 2.0.0
import functools
from typing import Optional

from torch.utils.data import DataLoader

from pytorch_lightning import LightningDataModule


# Modified datamodule
class Datamodule1(LightningDataModule):
    def __init__(self, _config):
        super().__init__()

        self.data_dir = _config["root_dataset"]

        self.num_workers = _config["num_workers"]
        self.batch_size = _config["per_gpu_batchsize"]
        self.eval_batch_size = self.batch_size

        self.draw_false_grid = _config["draw_false_grid"]
        self.img_size = _config["img_size"]
        self.downstream = _config["downstream"]

        self.nbr_fea_len = _config["nbr_fea_len"]

        self.tasks = [k for k, v in _config["loss_names"].items() if v >= 1]

    @property
    def dataset_cls(self):
        return Dataset1

    def set_train_dataset(self):
        self.train_dataset = self.dataset_cls(
            self.data_dir,
            split="train",
            draw_false_grid=self.draw_false_grid,
            downstream=self.downstream,
            nbr_fea_len=self.nbr_fea_len,
            tasks=self.tasks,
        )

    def set_val_dataset(self):
        self.val_dataset = self.dataset_cls(
            self.data_dir,
            split="val",
            draw_false_grid=self.draw_false_grid,
            downstream=self.downstream,
            nbr_fea_len=self.nbr_fea_len,
            tasks=self.tasks,
        )

    def set_test_dataset(self):
        self.test_dataset = self.dataset_cls(
            self.data_dir,
            split="test",
            draw_false_grid=self.draw_false_grid,
            downstream=self.downstream,
            nbr_fea_len=self.nbr_fea_len,
            tasks=self.tasks,
        )

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.set_train_dataset()
            self.set_val_dataset()

        if stage in (None, "test"):
            self.set_test_dataset()

        self.collate = functools.partial(
            self.dataset_cls.collate,
            img_size=self.img_size,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate,
        )


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import mean_absolute_error


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


# modified loss function with uncertainty logarithm
def compute_loss(logits, labels, log_sigma):
    loss = 0.5 * torch.exp(-log_sigma) * (labels - logits) ** 2 + 0.5 * log_sigma
    return loss.mean()


def compute_regression(pl_module, batch, normalizer):
    infer = pl_module.infer(batch)

    out = pl_module.regression_head(infer["cls_feats"])
    logits = out[:, 0]  # [B]
    log_sigma = out[:, 1]  # uncertainty logarithm
    labels = torch.FloatTensor(batch["target"]).to(logits.device)  # [B]
    assert len(labels.shape) == 1

    # normalize encode if config["mean"] and config["std], else pass
    labels = normalizer.encode(labels)
    loss = compute_loss(logits, labels, log_sigma)

    labels = labels.to(torch.float32)
    logits = logits.to(torch.float32)

    ret = {
        "cif_id": infer["cif_id"],
        "cls_feats": infer["cls_feats"],
        "regression_loss": loss,
        "regression_logits": normalizer.decode(logits),
        "regression_labels": normalizer.decode(labels),
        "log_sigma": log_sigma,  # uncertainty logarithm
    }

    # call update() loss and acc
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_regression_loss")(ret["regression_loss"])
    mae = getattr(pl_module, f"{phase}_regression_mae")(
        mean_absolute_error(
            ret["regression_logits"].cpu(), ret["regression_labels"].cpu()
        )
    )

    if pl_module.write_log:
        pl_module.log(f"regression/{phase}/loss", loss, sync_dist=True)
        pl_module.log(f"regression/{phase}/mae", mae, sync_dist=True)

    return ret


def compute_classification(pl_module, batch):
    infer = pl_module.infer(batch)

    logits, binary = pl_module.classification_head(
        infer["cls_feats"]
    )  # [B, output_dim]
    labels = torch.LongTensor(batch["target"]).to(logits.device)  # [B]
    assert len(labels.shape) == 1
    if binary:
        logits = logits.squeeze(dim=-1)
        loss = F.binary_cross_entropy_with_logits(input=logits, target=labels.float())
    else:
        loss = F.cross_entropy(logits, labels)

    ret = {
        "cif_id": infer["cif_id"],
        "cls_feats": infer["cls_feats"],
        "classification_loss": loss,
        "classification_logits": logits,
        "classification_labels": labels,
    }

    # call update() loss and acc
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_classification_loss")(
        ret["classification_loss"]
    )
    acc = getattr(pl_module, f"{phase}_classification_accuracy")(
        torch.sigmoid(ret["classification_logits"]),
        ret["classification_labels"],  # у авторов не было сигмоиды, но она нужна
    )

    if pl_module.write_log:
        pl_module.log(f"classification/{phase}/loss", loss, sync_dist=True)
        pl_module.log(f"classification/{phase}/accuracy", acc, sync_dist=True)

    return ret


def compute_mpp(pl_module, batch):
    infer = pl_module.infer(batch, mask_grid=True)

    mpp_logits = pl_module.mpp_head(infer["grid_feats"])  # [B, max_image_len+2, bins]
    mpp_logits = mpp_logits[
        :, :-1, :
    ]  # ignore volume embedding, [B, max_image_len+1, bins]
    mpp_labels = infer["grid_labels"]  # [B, max_image_len+1, C=1]

    mask = mpp_labels != -100.0  # [B, max_image_len, 1]

    # masking
    mpp_logits = mpp_logits[mask.squeeze(-1)]  # [mask, bins]
    mpp_labels = mpp_labels[mask].long()  # [mask]

    mpp_loss = F.cross_entropy(mpp_logits, mpp_labels)

    ret = {
        "mpp_loss": mpp_loss,
        "mpp_logits": mpp_logits,
        "mpp_labels": mpp_labels,
    }

    # call update() loss and acc
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mpp_loss")(ret["mpp_loss"])
    acc = getattr(pl_module, f"{phase}_mpp_accuracy")(
        ret["mpp_logits"], ret["mpp_labels"]
    )

    if pl_module.write_log:
        pl_module.log(f"mpp/{phase}/loss", loss, sync_dist=True)
        pl_module.log(f"mpp/{phase}/accuracy", acc, sync_dist=True)

    return ret


def compute_mtp(pl_module, batch):
    infer = pl_module.infer(batch)
    mtp_logits = pl_module.mtp_head(infer["cls_feats"])  # [B, hid_dim]
    mtp_labels = torch.LongTensor(batch["mtp"]).to(mtp_logits.device)  # [B]

    mtp_loss = F.cross_entropy(mtp_logits, mtp_labels)  # [B]

    ret = {
        "mtp_loss": mtp_loss,
        "mtp_logits": mtp_logits,
        "mtp_labels": mtp_labels,
    }

    # call update() loss and acc
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mtp_loss")(ret["mtp_loss"])
    acc = getattr(pl_module, f"{phase}_mtp_accuracy")(
        ret["mtp_logits"], ret["mtp_labels"]
    )

    if pl_module.write_log:
        pl_module.log(f"mtp/{phase}/loss", loss, sync_dist=True)
        pl_module.log(f"mtp/{phase}/accuracy", acc, sync_dist=True)

    return ret


def compute_vfp(pl_module, batch):
    infer = pl_module.infer(batch)

    vfp_logits = pl_module.vfp_head(infer["cls_feats"]).squeeze(-1)  # [B]
    vfp_labels = torch.FloatTensor(batch["vfp"]).to(vfp_logits.device)

    assert len(vfp_labels.shape) == 1

    vfp_loss = F.mse_loss(vfp_logits, vfp_labels)
    ret = {
        "vfp_loss": vfp_loss,
        "vfp_logits": vfp_logits,
        "vfp_labels": vfp_labels,
    }

    # call update() loss and acc
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_vfp_loss")(ret["vfp_loss"])
    mae = getattr(pl_module, f"{phase}_vfp_mae")(
        mean_absolute_error(ret["vfp_logits"], ret["vfp_labels"])
    )

    if pl_module.write_log:
        pl_module.log(f"vfp/{phase}/loss", loss, sync_dist=True)
        pl_module.log(f"vfp/{phase}/mae", mae, sync_dist=True)

    return ret


def compute_ggm(pl_module, batch):
    pos_len = len(batch["grid"]) // 2
    neg_len = len(batch["grid"]) - pos_len
    ggm_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len)]).to(
        pl_module.device
    )

    ggm_images = []
    for i, (bti, bfi) in enumerate(zip(batch["grid"], batch["false_grid"])):
        if ggm_labels[i] == 1:
            ggm_images.append(bti)
        else:
            ggm_images.append(bfi)

    ggm_images = torch.stack(ggm_images, dim=0)

    batch = {k: v for k, v in batch.items()}
    batch["grid"] = ggm_images

    infer = pl_module.infer(batch)
    ggm_logits = pl_module.ggm_head(infer["cls_feats"])  # cls_feats
    ggm_loss = F.cross_entropy(ggm_logits, ggm_labels.long())

    ret = {
        "ggm_loss": ggm_loss,
        "ggm_logits": ggm_logits,
        "ggm_labels": ggm_labels,
    }

    # call update() loss and acc
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_ggm_loss")(ret["ggm_loss"])
    acc = getattr(pl_module, f"{phase}_ggm_accuracy")(
        ret["ggm_logits"], ret["ggm_labels"]
    )

    if pl_module.write_log:
        pl_module.log(f"ggm/{phase}/loss", loss, sync_dist=True)
        pl_module.log(f"ggm/{phase}/accuracy", acc, sync_dist=True)

    return ret


def compute_moc(pl_module, batch):
    if "bbc" in batch.keys():
        task = "bbc"
    else:
        task = "moc"

    infer = pl_module.infer(batch)
    moc_logits = pl_module.moc_head(
        infer["graph_feats"][:, 1:, :]
    ).flatten()  # [B, max_graph_len] -> [B * max_graph_len]
    moc_labels = (
        infer["mo_labels"].to(moc_logits).flatten()
    )  # [B, max_graph_len] -> [B * max_graph_len]
    mask = moc_labels != -100

    moc_loss = F.binary_cross_entropy_with_logits(
        input=moc_logits[mask], target=moc_labels[mask]
    )  # [B * max_graph_len]

    ret = {
        "moc_loss": moc_loss,
        "moc_logits": moc_logits,
        "moc_labels": moc_labels,
    }

    # call update() loss and acc
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_{task}_loss")(ret["moc_loss"])
    acc = getattr(pl_module, f"{phase}_{task}_accuracy")(
        nn.Sigmoid()(ret["moc_logits"]), ret["moc_labels"].long()
    )

    if pl_module.write_log:
        pl_module.log(f"{task}/{phase}/loss", loss, sync_dist=True)
        pl_module.log(f"{task}/{phase}/accuracy", acc, sync_dist=True)

    return ret


# Regression head
from torch import nn
import torch.nn.functional as F


class RegressionHead(nn.Module):
    """
    Modified head for Regression
    Original: self.layer = nn.Linear(hid_dim, 1)
    """

    def __init__(self, hid_dim):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(hid_dim, 2048), nn.ReLU(), nn.Linear(2048, 2)
        )

    def forward(self, x):
        x = self.layer(x)
        return x


# Classification head (original MOFTransformer)
class ClassificationHead(nn.Module):
    """
    head for Classification
    """

    def __init__(self, hid_dim, n_classes):
        super().__init__()

        if n_classes == 2:
            self.layer = nn.Sequential(
                nn.Linear(hid_dim, 1),
            )
            self.binary = True
        else:
            self.layer = nn.Sequential(nn.Linear(hid_dim, 1))
            self.binary = False

    def forward(self, x):
        x = self.layer(x)

        return x, self.binary


# MOFTransformer version 2.1.0
from typing import Any, List
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import LightningModule

from moftransformer.modules import objectives, heads, module_utils
from moftransformer.modules.cgcnn import GraphEmbeddings
from moftransformer.modules.vision_transformer_3d import VisionTransformer3D

from moftransformer.modules.module_utils import Normalizer

import numpy as np
from sklearn.metrics import r2_score


# Modified model
class Module1(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.max_grid_len = config["max_grid_len"]
        self.vis = config["visualize"]

        # graph embedding with_unique_atoms
        self.graph_embeddings = GraphEmbeddings(
            atom_fea_len=config["atom_fea_len"],
            nbr_fea_len=config["nbr_fea_len"],
            max_graph_len=config["max_graph_len"],
            hid_dim=config["hid_dim"],
            vis=config["visualize"],
            n_conv=3,
        )
        self.graph_embeddings.apply(objectives.init_weights)

        # token type embeddings
        self.token_type_embeddings = nn.Embedding(2, config["hid_dim"])
        self.token_type_embeddings.apply(objectives.init_weights)

        # set transformer
        self.transformer = VisionTransformer3D(
            img_size=config["img_size"],
            patch_size=config["patch_size"],
            in_chans=config["in_chans"],
            embed_dim=config["hid_dim"],
            depth=config["num_layers"],
            num_heads=config["num_heads"],
            mlp_ratio=config["mlp_ratio"],
            drop_rate=config["drop_rate"],
            mpp_ratio=config["mpp_ratio"],
        )

        # class token
        self.cls_embeddings = nn.Linear(1, config["hid_dim"])
        self.cls_embeddings.apply(objectives.init_weights)

        # volume token
        self.volume_embeddings = nn.Linear(1, config["hid_dim"])
        self.volume_embeddings.apply(objectives.init_weights)

        # add features tokens
        self.add_features_embeddings = nn.Linear(
            N_ADD_FEATURES, config["hid_dim"]
        )  # linear embedding for add features
        self.add_features_embeddings.apply(objectives.init_weights)

        self.pooler = heads.Pooler(config["hid_dim"])
        self.pooler.apply(objectives.init_weights)

        # ===================== loss =====================
        if config["loss_names"]["ggm"] > 0:
            self.ggm_head = heads.GGMHead(config["hid_dim"])
            self.ggm_head.apply(objectives.init_weights)

        if config["loss_names"]["mpp"] > 0:
            self.mpp_head = heads.MPPHead(config["hid_dim"])
            self.mpp_head.apply(objectives.init_weights)

        if config["loss_names"]["mtp"] > 0:
            self.mtp_head = heads.MTPHead(config["hid_dim"])
            self.mtp_head.apply(objectives.init_weights)

        if config["loss_names"]["vfp"] > 0:
            self.vfp_head = heads.VFPHead(config["hid_dim"])
            self.vfp_head.apply(objectives.init_weights)

        if config["loss_names"]["moc"] > 0 or config["loss_names"]["bbc"] > 0:
            self.moc_head = heads.MOCHead(config["hid_dim"])
            self.moc_head.apply(objectives.init_weights)

        # ===================== Downstream =====================
        hid_dim = config["hid_dim"]

        if config["load_path"] != "" and not config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)
            print(f"load model : {config['load_path']}")

        if self.hparams.config["loss_names"]["regression"] > 0:
            self.regression_head = RegressionHead(hid_dim)
            self.regression_head.apply(objectives.init_weights)
            # normalization
            self.mean = config["mean"]
            self.std = config["std"]

        if self.hparams.config["loss_names"]["classification"] > 0:
            n_classes = config["n_classes"]
            self.classification_head = heads.ClassificationHead(hid_dim, n_classes)
            self.classification_head.apply(objectives.init_weights)

        module_utils.set_metrics(self)
        self.current_tasks = list()
        # ===================== load downstream (test_only) ======================

        if config["load_path"] != "" and config["test_only"]:
            ckpt = torch.load(config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)
            print(f"load model : {config['load_path']}")

        self.test_logits = []
        self.test_labels = []
        self.test_cifid = []
        self.write_log = True

    def infer(
        self,
        batch,
        mask_grid=False,
    ):
        cif_id = batch["cif_id"]
        atom_num = batch["atom_num"]  # [N']
        nbr_idx = batch["nbr_idx"]  # [N', M]
        nbr_fea = batch["nbr_fea"]  # [N', M, nbr_fea_len]
        crystal_atom_idx = batch["crystal_atom_idx"]  # list [B]
        uni_idx = batch["uni_idx"]  # list [B]
        uni_count = batch["uni_count"]  # list [B]

        grid = batch["grid"]  # [B, C, H, W, D]
        volume = batch["volume"]  # list [B]
        add_features = batch["add_features"]  # add features [B, N_ADD_FEATURES]

        if "moc" in batch.keys():
            moc = batch["moc"]  # [B]
        elif "bbc" in batch.keys():
            moc = batch["bbc"]  # [B]
        else:
            moc = None

        # get graph embeds
        (
            graph_embeds,  # [B, max_graph_len, hid_dim],
            graph_masks,  # [B, max_graph_len],
            mo_labels,  # if moc: [B, max_graph_len], else: None
        ) = self.graph_embeddings(
            atom_num=atom_num,
            nbr_idx=nbr_idx,
            nbr_fea=nbr_fea,
            crystal_atom_idx=crystal_atom_idx,
            uni_idx=uni_idx,
            uni_count=uni_count,
            moc=moc,
        )
        # add class embeds to graph_embeds
        cls_tokens = torch.zeros(len(crystal_atom_idx)).to(graph_embeds)  # [B]
        cls_embeds = self.cls_embeddings(cls_tokens[:, None, None])  # [B, 1, hid_dim]
        cls_mask = torch.ones(len(crystal_atom_idx), 1).to(graph_masks)  # [B, 1]

        graph_embeds = torch.cat(
            [cls_embeds, graph_embeds], dim=1
        )  # [B, max_graph_len+1, hid_dim]
        graph_masks = torch.cat([cls_mask, graph_masks], dim=1)  # [B, max_graph_len+1]

        # get grid embeds
        (
            grid_embeds,  # [B, max_grid_len+1, hid_dim]
            grid_masks,  # [B, max_grid_len+1]
            grid_labels,  # [B, grid+1, C] if mask_image == True
        ) = self.transformer.visual_embed(
            grid,
            max_image_len=self.max_grid_len,
            mask_it=mask_grid,
        )

        # add volume embeds to grid_embeds
        volume = torch.FloatTensor(volume).to(grid_embeds)  # [B]
        volume_embeds = self.volume_embeddings(volume[:, None, None])  # [B, 1, hid_dim]
        volume_mask = torch.ones(volume.shape[0], 1).to(grid_masks)
        # add add_features embeds to grid_embeds
        add_features = torch.FloatTensor(add_features).to(grid_embeds)  # [B]
        add_features_embeds = self.add_features_embeddings(
            add_features[:, None]
        )  # [B, 1, hid_dim]
        add_features_mask = torch.ones(add_features.shape[0], 1).to(grid_masks)
        grid_embeds = torch.cat(
            [grid_embeds, volume_embeds, add_features_embeds], dim=1
        )  # [B, max_grid_len+2, hid_dim]
        grid_masks = torch.cat(
            [grid_masks, volume_mask, add_features_mask], dim=1
        )  # [B, max_grid_len+2]

        # add token_type_embeddings
        graph_embeds = graph_embeds + self.token_type_embeddings(
            torch.zeros_like(graph_masks, device=self.device).long()
        )
        grid_embeds = grid_embeds + self.token_type_embeddings(
            torch.ones_like(grid_masks, device=self.device).long()
        )

        co_embeds = torch.cat(
            [graph_embeds, grid_embeds], dim=1
        )  # [B, final_max_len, hid_dim]
        co_masks = torch.cat(
            [graph_masks, grid_masks], dim=1
        )  # [B, final_max_len, hid_dim]

        x = co_embeds

        attn_weights = []
        for i, blk in enumerate(self.transformer.blocks):
            x, _attn = blk(x, mask=co_masks)

            if self.vis:
                attn_weights.append(_attn)

        x = self.transformer.norm(x)
        graph_feats, grid_feats = (
            x[:, : graph_embeds.shape[1]],
            x[:, graph_embeds.shape[1] :],
        )  # [B, max_graph_len, hid_dim], [B, max_grid_len+2, hid_dim]

        cls_feats = self.pooler(x)  # [B, hid_dim]

        ret = {
            "graph_feats": graph_feats,
            "grid_feats": grid_feats,
            "cls_feats": cls_feats,
            "raw_cls_feats": x[:, 0],
            "graph_masks": graph_masks,
            "grid_masks": grid_masks,
            "grid_labels": grid_labels,  # if MPP, else None
            "mo_labels": mo_labels,  # if MOC, else None
            "cif_id": cif_id,
            "attn_weights": attn_weights,
            "add_features": torch.tensor(batch["add_features"]).to(
                cls_feats
            ),  # add features
        }

        return ret

    def forward(self, batch):
        ret = dict()

        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Masked Patch Prediction
        if "mpp" in self.current_tasks:
            ret.update(objectives.compute_mpp(self, batch))

        # Graph Grid Matching
        if "ggm" in self.current_tasks:
            ret.update(objectives.compute_ggm(self, batch))

        # MOF Topology Prediction
        if "mtp" in self.current_tasks:
            ret.update(objectives.compute_mtp(self, batch))

        # Void Fraction Prediction
        if "vfp" in self.current_tasks:
            ret.update(objectives.compute_vfp(self, batch))

        # Metal Organic Classification (or Building Block Classfication)
        if "moc" in self.current_tasks or "bbc" in self.current_tasks:
            ret.update(objectives.compute_moc(self, batch))

        # regression
        if "regression" in self.current_tasks:
            normalizer = Normalizer(self.mean, self.std)
            # ret.update(objectives.compute_regression(self, batch, normalizer))
            ret.update(compute_regression(self, batch, normalizer))

        # classification
        if "classification" in self.current_tasks:
            # ret.update(objectives.compute_classification(self, batch))
            ret.update(compute_classification(self, batch))
        return ret

    def on_train_start(self):
        module_utils.set_task(self)
        self.write_log = True

    def training_step(self, batch, batch_idx):
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])
        return total_loss

    def on_train_epoch_end(self):
        module_utils.epoch_wrapup(self)

    def on_validation_start(self):
        module_utils.set_task(self)
        self.write_log = True

    def validation_step(self, batch, batch_idx):
        output = self(batch)

    def on_validation_epoch_end(self) -> None:
        module_utils.epoch_wrapup(self)

    def on_test_start(
        self,
    ):
        module_utils.set_task(self)

    def test_step(self, batch, batch_idx):
        output = self(batch)
        output = {
            k: (v.cpu() if torch.is_tensor(v) else v) for k, v in output.items()
        }  # update cpu for memory

        if "regression_logits" in output.keys():
            self.test_logits += output["regression_logits"].tolist()
            self.test_labels += output["regression_labels"].tolist()
        return output

    def on_test_epoch_end(self):
        module_utils.epoch_wrapup(self)

        # calculate r2 score when regression
        if len(self.test_logits) > 1:
            r2 = r2_score(np.array(self.test_labels), np.array(self.test_logits))
            self.log(f"test/r2_score", r2, sync_dist=True)
            self.test_labels.clear()
            self.test_logits.clear()

    def configure_optimizers(self):
        return module_utils.set_schedule(self)

    def on_predict_start(self):
        self.write_log = False
        module_utils.set_task(self)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        output = self(batch)

        if "classification_logits" in output:
            if self.hparams.config["n_classes"] == 2:
                output["classification_logits_index"] = torch.round(
                    torch.sigmoid(output["classification_logits"])
                ).to(
                    torch.int
                )  # added sigmoid
            else:
                softmax = torch.nn.Softmax(dim=1)
                output["classification_logits"] = softmax(
                    output["classification_logits"]
                )
                output["classification_logits_index"] = torch.argmax(
                    output["classification_logits"], dim=1
                )

        output = {
            k: (v.cpu().tolist() if torch.is_tensor(v) else v)
            for k, v in output.items()
            if ("logits" in k)
            or ("labels" in k)
            or ("sigma" in k)
            or "cif_id" == k  # added output of sigma
        }

        return output

    def on_predict_epoch_end(self, *args):
        self.test_labels.clear()
        self.test_logits.clear()

    def on_predict_end(
        self,
    ):
        self.write_log = True

    def lr_scheduler_step(self, scheduler, *args):
        if len(args) == 2:
            optimizer_idx, metric = args
        elif len(args) == 1:
            (metric,) = args
        else:
            raise ValueError(
                "lr_scheduler_step must have metric and optimizer_idx(optional)"
            )

        if pl.__version__ >= "2.0.0":
            scheduler.step(epoch=self.current_epoch)
        else:
            scheduler.step()


import sys
import os
import copy
import warnings
from pathlib import Path
import shutil

import pytorch_lightning as pl

from moftransformer.config import ex
from moftransformer.config import config as _config
from moftransformer.utils.validation import (
    get_valid_config,
    get_num_devices,
    ConfigurationError,
)

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)


_IS_INTERACTIVE = hasattr(sys, "ps1")


def run(root_dataset, downstream=None, log_dir="logs/", *, test_only=False, **kwargs):

    config = copy.deepcopy(_config())
    for key in kwargs.keys():
        if key not in config:
            raise ConfigurationError(f"{key} is not in configuration.")

    config.update(kwargs)
    config["root_dataset"] = root_dataset
    config["downstream"] = downstream
    config["log_dir"] = log_dir
    config["test_only"] = test_only

    main1(config)


# @ex.automain
def main1(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    _config = get_valid_config(_config)
    dm = Datamodule1(_config)
    model = Module1(_config)

    exp_name = f"{_config['exp_name']}"

    os.makedirs(_config["log_dir"], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val/the_metric",
        mode="max",
        save_last=True,
    )

    if _config["test_only"]:
        name = f'test_{exp_name}_seed{_config["seed"]}_from_{str(_config["load_path"]).split("/")[-1][:-5]}'
    else:
        name = f'{exp_name}_seed{_config["seed"]}_from_{str(_config["load_path"]).split("/")[-1][:-5]}'

    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=name,
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    num_device = get_num_devices(_config)
    print("num_device", num_device)

    # gradient accumulation
    if num_device == 0:
        accumulate_grad_batches = _config["batch_size"] // (
            _config["per_gpu_batchsize"] * _config["num_nodes"]
        )
    else:
        accumulate_grad_batches = _config["batch_size"] // (
            _config["per_gpu_batchsize"] * num_device * _config["num_nodes"]
        )

    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None

    if _IS_INTERACTIVE:
        strategy = None
    elif pl.__version__ >= "2.0.0":
        strategy = "ddp_find_unused_parameters_true"
    else:
        strategy = "ddp"

    log_every_n_steps = 10

    trainer = pl.Trainer(
        accelerator=_config["accelerator"],
        devices=_config["devices"],
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        strategy=strategy,
        benchmark=True,
        max_epochs=_config["max_epochs"],
        max_steps=max_steps,
        callbacks=callbacks,
        logger=logger,
        accumulate_grad_batches=accumulate_grad_batches,
        log_every_n_steps=log_every_n_steps,
        val_check_interval=_config["val_check_interval"],
        deterministic=True,
    )

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm, ckpt_path=_config["resume_from"])
        log_dir = Path(logger.log_dir) / "checkpoints"
        if best_model := next(log_dir.glob("epoch=*.ckpt")):
            shutil.copy(best_model, log_dir / "best.ckpt")

    else:
        trainer.test(model, datamodule=dm)


import sys
import os
import copy
import warnings
from pathlib import Path
import re
import csv

import pytorch_lightning as pl

from moftransformer.config import ex
from moftransformer.config import config as _config
from moftransformer.modules.module_utils import set_task
from moftransformer.utils.validation import (
    get_valid_config,
    get_num_devices,
    ConfigurationError,
    _IS_INTERACTIVE,
)

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)


def predict(
    root_dataset, load_path, downstream=None, split="all", save_dir=None, **kwargs
):

    config = copy.deepcopy(_config())
    for key in kwargs.keys():
        if key not in config:
            raise ConfigurationError(f"{key} is not in configuration.")

    config.update(kwargs)
    config["root_dataset"] = root_dataset
    config["downstream"] = downstream
    config["load_path"] = load_path
    config["test_only"] = True
    config["visualize"] = False
    config["split"] = split
    config["save_dir"] = save_dir

    return main_(config)


# @ex.automain
def main_(_config):
    config = copy.deepcopy(_config)

    config["test_only"] = True
    config["visualize"] = False

    os.makedirs(config["log_dir"], exist_ok=True)
    pl.seed_everything(config["seed"])

    num_device = get_num_devices(config)
    num_nodes = config["num_nodes"]
    if num_nodes > 1:
        warnings.warn(
            f"function <predict> only support 1 devices. change num_nodes {num_nodes} -> 1"
        )
        config["num_nodes"] = 1
    if num_device > 1:
        warnings.warn(
            f"function <predict> only support 1 devices. change num_devices {num_device} -> 1"
        )
        config["devices"] = 1

    config = get_valid_config(config)  # valid config
    model = Module1(config)
    dm = Datamodule1(config)
    model.eval()

    if _IS_INTERACTIVE:
        strategy = None
    elif pl.__version__ >= "2.0.0":
        strategy = "ddp_find_unused_parameters_true"
    else:
        strategy = "ddp"

    trainer = pl.Trainer(
        accelerator=config["accelerator"],
        devices=config["devices"],
        num_nodes=config["num_nodes"],
        precision=config["precision"],
        strategy=strategy,
        benchmark=True,
        max_epochs=1,
        log_every_n_steps=0,
        deterministic=True,
        logger=False,
    )

    # refine split
    split = config.get("split", "all")
    if split == "all":
        split = ["train", "val", "test"]
    elif isinstance(split, str):
        split = re.split(r",\s?", split)

    if split == ["test"]:
        dm.setup("test")
    elif "test" not in split:
        dm.setup("fit")
    else:
        dm.setup()

    # save_dir
    save_dir = config.get("save_dir", None)
    if save_dir is None:
        save_dir = Path(config["load_path"]).parent.parent
    else:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)

    # predict
    for s in split:
        if not s in ["train", "test", "val"]:
            raise ValueError(f"split must be train, test, or val, not {s}")

        savefile = save_dir / f"{s}_prediction.csv"
        dataloader = getattr(dm, f"{s}_dataloader")()
        rets = trainer.predict(model, dataloader)
        write_output(rets, savefile)

    print(f"All prediction values are saved in {save_dir}")
    return rets


def write_output(rets, savefile):
    keys = rets[0].keys()

    with open(savefile, "w") as f:
        wr = csv.writer(f)
        wr.writerow(keys)
        for ret in rets:
            if ret.keys() != keys:
                raise ValueError(ret.keys(), keys)

            for data in zip(*ret.values()):
                wr.writerow(data)
