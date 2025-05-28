########################################################
#                                                      #
#       author: omitted for anonymous submission       #
#                                                      #
#     credits and copyright coming upon publication    #
#                                                      #
########################################################


import os
import sys
import json

import pandas as pd
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from pathlib import Path


class ContrastiveLoss(nn.Module):
    def __init__(self, n_classes=19, void_label=-1):
        super().__init__()
        self.n_classes = n_classes
        self.void_label = void_label

    def forward(self, emb_k, emb_q, labels, epoch, tau=0.1):
        """
        emb_k: the feature bank with the aggregated embeddings over the iterations
        emb_q: the embeddings for the current iteration
        labels: the correspondent class labels for each sample in emb_q
        """
        if epoch:
            total_loss = torch.tensor(0.0).cuda()
            assert (emb_q.shape[0] == labels.shape[0]), "mismatch on emb_q and labels shapes!"
            emb_k = F.normalize(emb_k, dim=-1)
            emb_q = F.normalize(emb_q, dim=1)
        
            for i, emb in enumerate(emb_q):
                label = labels[i]
                if not (self.void_label in label.unique() and len(label.unique()) == 1):
                    label[label == self.void_label] = self.n_classes
                    label_sq = torch.unique(label, return_inverse=True)[1]
                    oh_label = (F.one_hot(label_sq)).unsqueeze(-2)  # one hot labels
                    count = oh_label.view(-1, oh_label.shape[-1]).sum(
                        dim=0
                    )  # num of pixels per cl
                    pred = emb.permute(1, 2, 0).unsqueeze(-1)
                    oh_pred = (
                        pred * oh_label
                    )  # (H, W, Nc, Ncp) Ncp num classes present in the label
                    oh_pred_flatten = oh_pred.view(
                        oh_pred.shape[0] * oh_pred.shape[1],
                        oh_pred.shape[2],
                        oh_pred.shape[3],
                    )
                    res_raw = oh_pred_flatten.sum(dim=0) / count  # avg feat per class
                    res_new = (res_raw[~res_raw.isnan()]).view(
                        -1, self.n_classes
                    )  # filter out nans given by intermediate classes (present because of oh)
                    label_list = label.unique()
                    if self.n_classes in label_list:
                        label_list = label_list[:-1]
                        res_new = res_new[:-1, :]

                    # temperature-scaled cosine similarity
                    final = (res_new.cuda() @ emb_k.T.cuda()) / 0.1

                    loss = F.cross_entropy(final, label_list)
                    total_loss += loss

            return total_loss / emb_q.shape[0]

        return torch.tensor(0).cuda()


class OWLoss(nn.Module):
    def __init__(self, n_classes, hinged=True, delta=0.1, void_label=-1, save_dir=None, applied=True):
        super().__init__()
        self.n_classes = n_classes
        self.hinged = hinged
        self.void_label = void_label
        self.delta = delta
        self.count = torch.zeros(self.n_classes).cuda()  # count for class
        self.features = {i: torch.zeros(self.n_classes).cuda() for i in range(self.n_classes)}
        # See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        # for implementation of Welford Alg.
        self.ex = {i: torch.zeros(self.n_classes).cuda() for i in range(self.n_classes)}
        self.ex2 = {i: torch.zeros(self.n_classes).cuda() for i in range(self.n_classes)}
        self.var = {i: torch.zeros(self.n_classes).cuda() for i in range(self.n_classes)}

        self.criterion = torch.nn.L1Loss(reduction="none")

        self.previous_features = None
        self.previous_count = None
        self.epoch = 0
        self.applied = applied
        if save_dir is None:
            save_dir = Path.cwd()
        if save_dir is not None:
            save_dir = Path(save_dir).joinpath('monitor')
            save_dir.mkdir(parents=True, exist_ok=True)
            self.save_dir = save_dir

    @torch.no_grad()
    def cumulate(self, logits: torch.Tensor, sem_gt: torch.Tensor):
        sem_pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        gt_labels = torch.unique(sem_gt).tolist()
        logits_permuted = logits.permute(0, 2, 3, 1)
        for label in gt_labels:
            if label == self.void_label:
                continue
            sem_gt_current = sem_gt == label
            sem_pred_current = sem_pred == label
            tps_current = torch.logical_and(sem_gt_current, sem_pred_current)
            if tps_current.sum() == 0:
                continue
            logits_tps = logits_permuted[torch.where(tps_current == 1)]
            # max_values = logits_tps[:, label].unsqueeze(1)
            # logits_tps = logits_tps / max_values
            avg_mav = torch.mean(logits_tps, dim=0)
            n_tps = logits_tps.shape[0]
            # features is running mean for mav
            self.features[label] = (self.features[label] * self.count[label] + avg_mav * n_tps)

            # Logits by class for true positive labels
            self.ex[label] += (logits_tps).sum(dim=0)
            self.ex2[label] += ((logits_tps) ** 2).sum(dim=0)
            self.count[label] += n_tps
            self.features[label] /= self.count[label] + 1e-8

    def forward(self, logits: torch.Tensor, sem_gt: torch.Tensor, is_train: bool) -> torch.Tensor:
        if is_train:
            # update mav only at training time
            sem_gt = sem_gt#.type(torch.uint8)
            self.cumulate(logits, sem_gt)
        if self.previous_features == None:
            return torch.tensor(0.0).cuda()
        gt_labels = torch.unique(sem_gt).tolist()

        logits_permuted = logits.permute(0, 2, 3, 1)

        acc_loss = torch.tensor(0.0).cuda()
        if not self.applied:
            return acc_loss
        for label in gt_labels[1:]:
            # var_selection = self.var[label] > 1e-5
            # if var_selection.sum() == 0:
            #     continue
            mav = self.previous_features[label]
            logs = logits_permuted[torch.where(sem_gt == label)]
            mav = mav.expand(logs.shape[0], -1)
            if self.previous_count[label] > 0 and not self.var[label].sum() == 0:
                ew_l1 = self.criterion(logs, mav)
                # ew_l1 = ew_l1[:, var_selection] / (self.var[label][var_selection] + 1e-8)
                # Car variance [0:1]
                # We do this because the vairances become too small and the loss explodes
                filter_out_zero = self.var[label] > 0
                variance = self.var[label].clone()
                non_zero_min = variance[filter_out_zero].abs().min()
                variance[filter_out_zero] = non_zero_min
                norm_variance = variance / non_zero_min
                ew_l1 = ew_l1 / (norm_variance + 1e-8)
                if self.hinged:
                    ew_l1 = F.relu(ew_l1 - self.delta)
                ew_l1 = ew_l1.mean()
                if ew_l1.isnan().any():
                    print("NaN in ew_l1, skipping this label")
                    continue
                acc_loss += ew_l1

        return acc_loss

    def update(self):
        self.previous_features = self.features
        self.previous_count = self.count
        for c in self.var.keys():
            self.var[c] = (self.ex2[c] - self.ex[c] ** 2 / (self.count[c] + 1e-8)) / (self.count[c] + 1e-8)
            # Save to file with json
            with open(self.save_dir.joinpath(f"var_{self.epoch}.json"), "w") as f:
                var_to_save = {k: v.cpu().numpy().tolist() for k, v in self.var.items()}
                json.dump(var_to_save, f)
            with open(self.save_dir.joinpath(f"ex2_{self.epoch}.json"), "w") as f:
                ex2_to_save = {k: v.cpu().numpy().tolist() for k, v in self.ex2.items()}
                json.dump(ex2_to_save, f)
            with open(self.save_dir.joinpath(f"ex_{self.epoch}.json"), "w") as f:
                ex_to_save = {k: v.cpu().numpy().tolist() for k, v in self.ex.items()}
                json.dump(ex_to_save, f)
            with open(self.save_dir.joinpath(f"count_{self.epoch}.json"), "w") as f:
                count_to_save = self.count.cpu().numpy().tolist()
                json.dump(count_to_save, f)

        self.epoch += 1

        # resetting for next epoch
        self.count = torch.zeros(self.n_classes)  # count for class
        self.features = {i: torch.zeros(self.n_classes).cuda() for i in range(self.n_classes)}
        self.ex = {i: torch.zeros(self.n_classes).cuda() for i in range(self.n_classes)}
        self.ex2 = {i: torch.zeros(self.n_classes).cuda() for i in range(self.n_classes)}

        return self.previous_features, self.var

    def read(self):
        mav_tensor = torch.zeros(self.n_classes, self.n_classes)
        for key in self.previous_features.keys():
            mav_tensor[key] = self.previous_features[key]
        return mav_tensor


# This is training the model to do anomaly segmentation - it is being trained to differentiate between known and unknown classes
class ObjectosphereLoss(nn.Module):
    def __init__(self, sigma=1.0, void_label=-1):
        super().__init__()
        self.sigma = sigma
        self.void_label = void_label

    def forward(self, logits, sem_gt): # 0,1,2,3, 0, 2, 3, 1 B, C, H, W -> B, H, W, C
        logits_unk = logits.permute(0, 2, 3, 1)[torch.where(sem_gt == self.void_label)]
        logits_kn = logits.permute(0, 2, 3, 1)[torch.where(sem_gt != self.void_label)]

        if len(logits_unk):
            loss_unk = (torch.linalg.norm(logits_unk, dim=1)**2).mean()
        else:
            loss_unk = torch.tensor(0)
        if len(logits_kn):
            loss_kn = F.relu(self.sigma - (torch.linalg.norm(logits_kn, dim=1)**2)).mean()
        else:
            loss_kn = torch.tensor(0)

        loss = 10 * loss_unk + loss_kn
        return loss


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, device, weight, void_label=-1):
        super(CrossEntropyLoss2d, self).__init__()
        self.void_label = void_label
        self.weight = torch.tensor(weight).to(device)
        self.num_classes = len(self.weight) + 1  # +1 for void
        # if self.num_classes < 2**8:
        #     self.dtype = torch.uint8
        # else:
        #     self.dtype = torch.int16
        # IMPORTANT: ignore index -1 (the void)
        self.ce_loss = nn.CrossEntropyLoss(torch.from_numpy(np.array(weight)).float(), reduction="none", ignore_index=void_label)
        self.ce_loss.to(device)

    def forward(self, inputs, targets):
        # targets_m = targets.clone()
        if (targets == self.void_label).all(): # i.e. if all labels are void (-1)
            return [torch.tensor(0.0).cuda()]
            # import ipdb;ipdb.set_trace()  # fmt: skip
        # targets_m -= 1
        # assert (targets - 1 >= 0).all()
        # print("(targets-1).unique", (targets-1).unique())
        # print("inputs.shape", inputs.shape)
        # print(targets)
        # print(inputs)
        loss_all = self.ce_loss(inputs, targets.long())
        # if (self.weight == 1).all():
        divisor_weighted_pixel_sum = (targets >= 0).sum()
        # else:
            # number_of_pixels_per_class = torch.bincount(targets.flatten() - 1, minlength=self.num_classes)
            # divisor_weighted_pixel_sum = torch.sum(number_of_pixels_per_class[1:] * self.weight) # without void

        return [torch.sum(loss_all) / divisor_weighted_pixel_sum]


class CrossEntropyLoss2dForValidData:
    def __init__(self, device, weight, weighted_pixel_sum, void_label=-1):
        super(CrossEntropyLoss2dForValidData, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(torch.from_numpy(np.array(weight)).float(), reduction="sum", ignore_index=void_label)
        self.ce_loss.to(device)
        self.weighted_pixel_sum = weighted_pixel_sum
        self.total_loss = 0

    def add_loss_of_batch(self, inputs, targets):
        # targets_m = targets.clone()
        # targets_m -= 1
        loss = self.ce_loss(inputs, targets.long())
        self.total_loss += loss

    def compute_whole_loss(self):
        return self.total_loss.detach().cpu().numpy().item() / self.weighted_pixel_sum.item()

    def reset_loss(self):
        self.total_loss = 0


class CrossEntropyLoss2dForValidDataUnweighted:
    def __init__(self, device, void_label=-1):
        super(CrossEntropyLoss2dForValidDataUnweighted, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=None, reduction="sum", ignore_index=void_label)
        self.ce_loss.to(device)
        self.nr_pixels = 0
        self.total_loss = 0

    def add_loss_of_batch(self, inputs, targets):
        # targets_m = targets.clone()
        # targets_m -= 1
        loss = self.ce_loss(inputs, targets)
        self.total_loss += loss
        self.nr_pixels += torch.sum(targets >= 0) # only non void pixels

    def compute_whole_loss(self):
        return (
            self.total_loss.detach().cpu().numpy().item()
            / self.nr_pixels.detach().cpu().numpy().item()
        )

    def reset_loss(self):
        self.total_loss = 0
        self.nr_pixels = 0


def print_log(
    epoch, local_count, count_inter, dataset_size, loss, time_inter, learning_rates
):
    print_string = "Train Epoch: {:>3} [{:>4}/{:>4} ({: 5.1f}%)]".format(
        epoch, local_count, dataset_size, 100.0 * local_count / dataset_size
    )
    for i, lr in enumerate(learning_rates):
        print_string += "   lr_{}: {:>6}".format(i, round(lr, 10))
    print_string += "   Loss: {:0.6f}".format(loss.item())
    print_string += "  [{:0.2f}s every {:>4} data]".format(time_inter, count_inter)
    print(print_string, flush=True)


def save_ckpt(ckpt_dir, model, optimizer, epoch):
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    ckpt_model_filename = "ckpt_epoch_{}.pth".format(epoch)
    path = os.path.join(ckpt_dir, ckpt_model_filename)
    torch.save(state, path)
    print("{:>2} has been successfully saved".format(path))


def save_ckpt_every_epoch(
    ckpt_dir, model, optimizer, epoch, best_miou, best_miou_epoch, mavs, stds, ows_loss
):
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_miou": best_miou,
        "best_miou_epoch": best_miou_epoch,
        "mavs": mavs,
        "stds": stds,
        "ows_loss": ows_loss,
    }
    ckpt_model_filename = "ckpt_latest.pth".format(epoch)
    path = os.path.join(ckpt_dir, ckpt_model_filename)
    torch.save(state, path)


def load_ckpt(model, optimizer, model_file, device):
    if os.path.isfile(model_file):
        print("=> loading checkpoint '{}'".format(model_file))
        if device.type == "cuda":
            checkpoint = torch.load(model_file)
        else:
            checkpoint = torch.load(
                model_file, map_location=lambda storage, loc: storage
            )

        mav_dict = checkpoint["mavs"]
        std_dict = checkpoint["stds"]
        ows_loss = checkpoint["ows_loss"] if "ows_loss" in checkpoint else None

        model.load_state_dict(checkpoint["state_dict"])

        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer"])
        print(
            "=> loaded checkpoint '{}' (epoch {})".format(
                model_file, checkpoint["epoch"]
            )
        )
        epoch = checkpoint["epoch"]
        if "best_miou" in checkpoint:
            best_miou = checkpoint["best_miou"]
            print("Best mIoU:", best_miou)
        else:
            best_miou = 0

        if "best_miou_epoch" in checkpoint:
            best_miou_epoch = checkpoint["best_miou_epoch"]
            print("Best mIoU epoch:", best_miou_epoch)
        else:
            best_miou_epoch = 0
        return epoch, best_miou, best_miou_epoch, mav_dict, std_dict, ows_loss
    else:
        print("=> no checkpoint found at '{}'".format(model_file))
        sys.exit(1)


def get_best_checkpoint(ckpt_dir, key="mIoU_test"):
    ckpt_path = None
    log_file = os.path.join(ckpt_dir, "logs.csv")
    if os.path.exists(log_file):
        data = pd.read_csv(log_file)
        idx = data[key].idxmax()
        miou = data[key][idx]
        epoch = data.epoch[idx]
        ckpt_path = os.path.join(ckpt_dir, f"ckpt_epoch_{epoch}.pth")
    assert ckpt_path is not None, f"No trainings found at {ckpt_dir}"
    assert os.path.exists(ckpt_path), f"There is no weights file named {ckpt_path}"
    print(f"Best mIoU: {100*miou:0.2f} at epoch: {epoch}")
    return ckpt_path
