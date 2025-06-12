########################################################
#                                                      #
#       author: omitted for anonymous submission       #
#                                                      #
#     credits and copyright coming upon publication    #
#                                                      #
########################################################


import os
import sys

import pandas as pd
import torch


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
    print(f"Best mIoU: {100 * miou:0.2f} at epoch: {epoch}")
    return ckpt_path
